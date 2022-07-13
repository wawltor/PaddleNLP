# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import error
from ..utils.log import logger
from ..taskflow.utils import dygraph_mode_guard


class Predictor:

    def __init__(self,
                 model_path,
                 input_specs,
                 model_config_path=None,
                 precision='fp32',
                 device=None):
        self._model_path = model_path
        self._static_model_path = "auto_static"
        self._model_config_path = model_config_path
        self._input_specs = input_specs
        self._precision = 'fp32'
        self._cpu_thread = 8
        self._config = None

    def _model_config_path(self):
        if self._model_config_path is not None:
            return self._model_config_path
        else:
            for file_name in os.listdir(self._model_path):
                if file_name.count('model_config'):
                    return os.path.join(self._model_path, file_name)
            return None
        return None

    def _static_model_path(self):
        # The model path had the static_model_path
        static_model_path = os.path.join(self._model_path, self.auto_static,
                                         'inference.pdiparams')
        if os.path.exists(static_model_path):
            return os.path.join(self._model_path, "inference")
        for file_name in os.listdir(self._model_path):
            if file_name.count(".pdiparams"):
                return os.path.join(model_path, file_name[:-10])
        return None

    def create_predictor(self):
        # Get the model parameter path and model config path
        model_config_path = self._model_config_path()
        static_model_path = self._static_model_path()

        # Convert the Draph Model to Static Model
        if static_model_path is None:
            if os.path.exists(model_config_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        'model_config.json')
            config_json_dict = json.load(open(model_config_path))
            class_name = config_json_dict['init_class']
            # FIXME(wawltor) The model class is not class of paddlenlp.transformers
            model_class = getattr(paddlenlp.transformers, class_name)
            model_instance = model_class.from_pretrained(self._model_path)
            model_instance.eval()
            self._convert_dygraph_to_static(model_instance)
            static_model_path = os.path.join(self._model_path, 'inference')

        # Load the inference model and maybe we will convert the onnx model
        # Judge the predictor type for the inference
        predictor_type = self._check_predictor_type()

    def _get_model_precision_type(self):
        pass

    def _check_predictor_type(self):
        predictor_type = 'paddle_inference'
        if paddle.get_device() == 'cpu' and self._precision == 'fp16':
            logger.warning(
                "The inference precision is change to 'fp32', 'fp16' inference only takes effect on gpu."
            )
        else:
            if self._precision == 'fp16':
                try:
                    import onnx
                    import onnxruntime as ort
                    import paddle2onnx
                    from onnxconverter_common import float16
                    predictor_type = 'onnxruntime'
                except:
                    logger.warning(
                        "The inference precision is change to 'fp32', please install the dependencies that required for 'fp16' inference, pip install onnxruntime-gpu onnx onnxconverter-common"
                    )
        return predictor_type

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddele static mode. 
        """
        if paddle.get_device() == 'cpu':
            self._config.disable_gpu()
            self._config.enable_mkldnn()
        else:
            self._config.enable_use_gpu(100, self.kwargs['device_id'])
            # TODO(linjieccc): enable embedding_eltwise_layernorm_fuse_pass after fixed
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        self._config.set_cpu_math_library_num_threads(self._num_threads)
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = [
            self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        ]

    def _prepare_onnx_mode(self):
        import onnx
        import onnxruntime as ort
        import paddle2onnx
        from onnxconverter_common import float16
        onnx_dir = os.path.join(self._task_path, 'onnx')
        if not os.path.exists(onnx_dir):
            os.mkdir(onnx_dir)
        float_onnx_file = os.path.join(onnx_dir, 'model.onnx')
        if not os.path.exists(float_onnx_file):
            onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=self._static_model_file,
                params_file=self._static_params_file,
                opset_version=13,
                enable_onnx_checker=True)
            with open(float_onnx_file, "wb") as f:
                f.write(onnx_model)
        fp16_model_file = os.path.join(onnx_dir, 'fp16_model.onnx')
        if not os.path.exists(fp16_model_file):
            onnx_model = onnx.load_model(float_onnx_file)
            trans_model = float16.convert_float_to_float16(onnx_model,
                                                           keep_io_types=True)
            onnx.save_model(trans_model, fp16_model_file)
        providers = ['CUDAExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._num_threads
        sess_options.inter_op_num_threads = self._num_threads
        self.predictor = ort.InferenceSession(fp16_model_file,
                                              sess_options=sess_options,
                                              providers=providers)
        assert 'CUDAExecutionProvider' in self.predictor.get_providers(), f"The environment for GPU inference is not set properly. " \
            "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. " \
            "Please run the following commands to reinstall: \n " \
            "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"

    def _convert_dygraph_to_static(self, model_instance):
        """
        Convert the dygraph model to static model.
        """
        assert model_instance is not None, 'The dygraph model must be created before converting the dygraph model to static model.'
        assert self._input_specs is not None, 'The input spec must be created before converting the dygraph model to static model.'
        logger.info("Converting to the inference model cost a little time.")
        try:
            static_model = paddle.jit.to_static(model_instance,
                                                input_spec=self._input_spec)
            save_path = os.path.join(self._model_path, "static", "inference")
            paddle.jit.save(static_model, save_path)
            logger.info(
                "The inference model save in the path:{}".format(save_path))
        except:
            logger.warning(
                "Fail convert toe inference model, please create the issue for the developers,"
                "the issue link: https://github.com/PaddlePaddle/PaddleNLP/issues"
            )
            sys.exit(-1)
