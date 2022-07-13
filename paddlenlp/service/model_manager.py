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

import time
from .handlers import TaskflowHandler


class ModelManager:

    def __init__(self):
        self._task_name = 'models_predict'
        self._model_path = None

    def _register(self, task_name, model_path, handler=None, device=None):
        self._task_name = task_name
        self._model_path = model_path

    def _load_model_config():
        config_path = os.path.join(self._model_path, '')

    def _predict(self, text):
        t = time.time()
        t = int(round(t * 1000))
        task_index = t % len(self._task)
        return self._handler(self._task[task_index], text)
