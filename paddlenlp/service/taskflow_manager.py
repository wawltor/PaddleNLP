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


class TaskflowManager:
    """
    The TaskflowManager could predict the raw text.
    """

    def __init__(self):
        self._task = None
        self._handler_func = TaskflowHandler.process

    def _register(self, task, func=None):
        self._task = task
        if func is not None:
            self._handler = func

    def predict(self, text):
        t = time.time()
        t = int(round(t * 1000))
        task_index = t % len(self._task)
        return self._handler_func(self._task[task_index], text)
