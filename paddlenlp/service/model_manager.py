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
from .predictor import Predictor
from ..utils.tools import get_env_device
from ..transformers import AutoTokenizer


class ModelManager:

    def __init__(self, task_name, params_path, model_class_or_name,
                 tokenizer_name, input_spec, handler, precision, device_id,
                 batch_size):
        self._task_name = task_name
        self._params_path = params_path
        self._model_class_or_name = model_class_or_name
        self._tokenizer_name = tokenizer_name
        self._input_spec = input_spec
        self._handler = handler
        self._device_id = device_id

    def register(self):
        device = get_env_device()
        predictor_list = []
        if device == 'cpu' or self._device_id == -1:
            predictor = Predictor(self._params_path, model_class_or_name,
                                  input_spec, precision, 'cpu')
            predictor_list.append(predictor)
        elif isinstance(self._device_id, int):
            predictor = Predictor(self._params_path, model_class_or_name,
                                  input_spec, precision, 'gpu:0')
            predictor_list.append(predictor)
        elif isinstance(self._device_id, list):
            for device in device_id:
                predictor = Predictor(self._params_path, model_class_or_name,
                                      input_spec, precision,
                                      'gpu:' + str(device))
                predictor_list.append(predictor)
        self._predictor_list = predictor_list

        # Get tokenizer name
    def _get_tokenizer(self):
        if self._tokenizer_name is not None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_name)
        else:
            for file_name in os.list_dir(self._params_path):
                if file_name.count('tokenizer_config'):
                    tokenizer_config_path = os.path.join(
                        self._model_path, file_name)
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._params_path)
                    break

    def predict(self, text, text_pair=None):
        t = time.time()
        t = int(round(t * 1000))
        predictor_id = t % len(self._predictor_list)
        return self._handler(self.predictor_list[predictor_id], self._tokenizer,
                             text, text_pair)
