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
import errno
import json
import math
from multiprocessing import cpu_count
import paddle
from paddle import inference
from ..transformers import PretrainedModel
from ..utils.log import logger
from ..taskflow.utils import dygraph_mode_guard
from .input_mapping import mappings


class Predictor:

    def __init__(self, model_path, model_class_or_name, input_spec, precision,
                 device):
        self._model_path = model_path
        self._default_static_model_path = "auto_static"
        self._model_class_or_name = model_class_or_name
        self._input_spec = input_spec
        self._precision = precision
        self._cpu_thread = 8
        self._config = None
        self._device = device
        self._num_threads = math.ceil(cpu_count() / 2)
        paddle.set_device(device)
        self._create_predictor()

    def _model_class(self):
        if self._model_class_or_name is not None:
            if isinstance(self._model_class_or_name, str):
                return getattr(paddlenlp.transformers, class_name)
            elif isinstance(self._model_class_or_name, PretrainedModel):
                return self._model_class_or_name
            else:
                logger.error(
                    'The argrument of `model_class_or_name`  must be the name or class of paddlenlp.transformers.PretrainedModel'
                )
                sys.exit(-1)
        else:
            for file_name in os.listdir(self._model_path):
                if file_name.count('model_config'):
                    model_config_path = os.path.join(self._model_path,
                                                     file_name)
                    config_json_dict = json.load(open(model_config_path))
                    class_name = config_json_dict['init_class']
                    import paddlenlp
                    model_class = getattr(paddlenlp.transformers, class_name)
                    return model_class
        return None

    def _model_input_spec(self, model_class):
        if self._input_spec is not None:
            return self._input_spec
        model_name = model_class.__name__
        if model_name in mappings:
            return mappings[model_name][0]
        return None

    def _get_default_static_model_path(self):
        # The model path had the static_model_path
        static_model_path = os.path.join(self._model_path,
                                         self._default_static_model_path,
                                         'inference.pdiparams')
        if os.path.exists(static_model_path):
            return os.path.join(self._model_path,
                                self._default_static_model_path, "inference")
        for file_name in os.listdir(self._model_path):
            # FIXME(wawltor) The path maybe not correct
            if file_name.count(".pdiparams"):
                return os.path.join(model_path, file_name[:-10])
        return None

    def _is_int8_model(self, model_path):
        paddle.set_device('cpu')
        model = paddle.jit.load(model_path)
        program = model.program()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type.count("quantize"):
                    paddle.set_device(self._device)
                    return True
        paddle.set_device(self._device)
        return False

    def _create_predictor(self):
        # Get the model parameter path and model config path
        model_class = self._model_class()

        if model_class is None:
            logger.error(
                'The argrument of `model_class_or_name`  must be the name or class of paddlenlp.transformers.PretrainedModel'
            )
            sys.exit(-1)
        static_model_path = self._get_default_static_model_path()

        # Convert the Draph Model to Static Model
        is_from_static = True
        if static_model_path is None:
            # FIXME(wawltor) The model class is not class of paddlenlp.transformers
            model_instance = model_class.from_pretrained(self._model_path)
            model_instance.eval()
            input_spec = self._model_input_spec(model_class)
            if input_spec is None:
                logger.error('You must be set the input_spec.')
                sys.exit(-1)
            self._convert_dygraph_to_static(model_instance, input_spec)
            static_model_path = os.path.join(self._model_path,
                                             self._default_static_model_path,
                                             'inference')
            is_from_static = False

        is_int8_model = False
        if is_from_static:
            is_int8_model = self._is_int8_model(static_model_path)
        # Load the inference model and maybe we will convert the onnx model
        # Judge the predictor type for the inference
        if self._precision == 'int8' and not is_int8_model:
            self._precision = 'fp32'
        if is_int8_model:
            self._precision = 'int8'

        self._predictor_type = self._check_predictor_type()
        if self._predictor_type == 'paddle_inference':
            self._prepare_paddle_mode(static_model_path)
        else:
            self._prepare_onnx_mode()

    def _check_predictor_type(self):
        predictor_type = 'paddle_inference'
        device = paddle.get_device()
        if self._precision == 'int8' or device == "xpu" or device == "cpu":
            predictor_type = 'paddle_inference'
        else:
            if device == 'gpu' and self._precision == 'fp16':
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

    def _prepare_paddle_mode(self, static_model_path):
        """
        Construct the input data and predictor in the PaddlePaddele static mode. 
        """
        self._config = paddle.inference.Config(static_model_path + ".pdmodel",
                                               static_model_path + ".pdiparams")
        self._config.disable_glog_info()
        if paddle.get_device() == 'cpu':
            self._config.disable_gpu()
            self._config.enable_mkldnn()
            self._config.enable_memory_optim()
            if self._precision == 'int8':
                config.enable_mkldnn_bfloat16()
            elif self._precision == 'fp16':
                config.enable_mkldnn_int8()
        else:
            self._config.enable_use_gpu(100, int(self._device.split(":")[-1]))
            precision_type = inference.PrecisionType.Float32
            if self._precision == 'int8':
                precision_type = inference.PrecisionType.INT8
                # FIXME(wawltor) The paddlenlp serving support the int8 model
                logger.warning(
                    'The PaddleNLP serving do not support the INT8 model, we will support later!'
                )
                sys.exit(-1)

        self._config.switch_use_feed_fetch_ops(False)
        self._config.set_cpu_math_library_num_threads(self._num_threads)
        self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        self._predictor = paddle.inference.create_predictor(self._config)
        self._input_handles = [
            self._predictor.get_input_handle(name)
            for name in self._predictor.get_input_names()
        ]
        self._output_handles = [
            self._predictor.get_output_handle(name)
            for name in self._predictor.get_output_names()
        ]

    def _prepare_onnx_mode(self):
        import onnx
        import onnxruntime as ort
        import paddle2onnx
        from onnxconverter_common import float16
        onnx_dir = os.path.join(self._model_path, 'onnx')
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
        #providers = ['CUDAExecutionProvider', provider_options=[{'device_id': device_id}]]
        providers = ['CUDAExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = self._num_threads
        self._predictor = ort.InferenceSession(fp16_model_file,
                                               sess_options=sess_options,
                                               providers=providers,
                                               provider_options=[{
                                                   'device_id':
                                                   device_id
                                               }])
        assert 'CUDAExecutionProvider' in self._predictor.get_providers(), f"The environment for GPU inference is not set properly. " \
            "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. " \
            "Please run the following commands to reinstall: \n " \
            "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"

    def _convert_dygraph_to_static(self, model_instance, input_spec):
        """
        Convert the dygraph model to static model.
        """
        assert model_instance is not None, 'The dygraph model must be created before converting the dygraph model to static model.'
        assert input_spec is not None, 'The input spec must be created before converting the dygraph model to static model.'
        logger.info(
            "Converting to the static inference model will cost a little time, please do not break this process."
        )
        try:
            static_model = paddle.jit.to_static(model_instance,
                                                input_spec=input_spec)
            save_path = os.path.join(self._model_path,
                                     self._default_static_model_path,
                                     "inference")
            paddle.jit.save(static_model, save_path)
            logger.info("The static inference model save in the path:{}".format(
                save_path))
        except:
            logger.warning(
                "Fail convert to inference model, please create the issue for the developers,"
                "the issue link: https://github.com/PaddlePaddle/PaddleNLP/issues"
            )
            sys.exit(-1)
