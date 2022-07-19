# coding:utf-8
# copyright (c) 2022  paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license"
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import time
import json
import paddlenlp
from .predictor import Predictor
from .input_mapping import mappings
from ..utils.tools import get_env_device
from ..transformers import AutoTokenizer


class ModelManager:

    def __init__(self, task_name, model_path, model_class_or_name,
                 tokenizer_name, input_spec, handler, precision, device_id,
                 batch_size):
        self._task_name = task_name
        self._model_path = model_path
        self._model_class_or_name = model_class_or_name
        self._tokenizer_name = tokenizer_name
        self._input_spec = input_spec
        self._handler = handler
        self._precision = precision
        self._device_id = device_id
        self._tokenizer = None

    def register(self):
        # Get the model handler
        if self._handler is None:
            model_class = self._get_model_class()
            model_handler = self._get_model_handler_class(model_class)
            print("The model handler is {}".format(model_handler))
            self._handler = model_handler.process
            assert self._handler is not None, 'The Handler must be not register, you could set the class of handler'
        print(self._handler)

        # Create the model predictor
        device = get_env_device()
        predictor_list = []
        if device == 'cpu' or self._device_id == -1:
            predictor = Predictor(self._model_path, self._model_class_or_name,
                                  self._input_spec, self._precision, 'cpu')
            predictor_list.append(predictor)
        elif isinstance(self._device_id, int):
            predictor = Predictor(self._model_path, self._model_class_or_name,
                                  self._input_spec, self._precision,
                                  'gpu:' + str(self._device_id))
            predictor_list.append(predictor)
        elif isinstance(self._device_id, list):
            for device in device_id:
                predictor = Predictor(self._model_path,
                                      self._model_class_or_name,
                                      self._input_spec, self._precision,
                                      'gpu:' + str(device))
                predictor_list.append(predictor)
        self._predictor_list = predictor_list

        # Get the tokenize of model
        self._get_tokenizer()

    def _get_model_class(self):
        if self._model_class_or_name is not None:
            if isinstance(self._model_class_or_name, str):
                return getattr(paddlenlp.transformers,
                               self._model_class_or_name)
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
                    model_class = getattr(paddlenlp.transformers, class_name)
                    return model_class
        return None

    def _get_model_handler_class(self, model_class):
        model_name = model_class.__name__
        if model_name in mappings:
            return mappings[model_name][1]
        return None

    def _get_tokenizer(self):
        if self._tokenizer_name is not None:
            if isinstance(self._tokenizer_name, str):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_name)
            else:
                logger.error(
                    'The argrument of `tokenizer_name`  must be the name of tokenizer.'
                )
                sys.exit(-1)
        else:
            for file_name in os.listdir(self._model_path):
                if file_name.count('tokenizer_config'):
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._model_path)
                    break
        assert self._tokenizer is not None, 'The tokenizer must be not register, you could set the class of Tokenizer'

    def _get_predict_id(self):
        t = time.time()
        t = int(round(t * 1000))
        predictor_id = t % len(self._predictor_list)
        return predictor_id

    def predict(self, text, text_pair):
        predictor_id = self._get_predict_id()
        return self._handler(self._predictor_list[predictor_id],
                             self._tokenizer, text, text_pair)
