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
import typing
from typing import Any, List, Optional
import hashlib
from fastapi import APIRouter, Request
from ..base_router import BaseRouterManager

from pydantic import BaseModel, Extra
from pydantic import create_model


class ResponseBase(BaseModel):
    text: Optional[str] = None


class HttpRouterManager(BaseRouterManager):

    def register_models_router(self, task_name):
        # url path to register the model
        paths = [f"/models/{task_name}"]
        print(paths)

        # unique name to create the pydantic model
        unique_name = (hashlib.md5(task_name.encode()).hexdigest())

        # create response model
        resp_model = create_model(
            "V1V1ResponseModel" + unique_name,
            result=(typing.Any, ...),
            __base__=ResponseBase,
        )

        # template predict endpoint function to dynamically serve different models
        def predict(
            request: Request,
            text: str,
            text_pair: str = None,
        ):
            print("The text:{}, text_pair:{}".format(text, text_pair))
            result = self._app._model_manager.predict(text, text_pair)
            return {"text": text, 'result': result}

        # register the route and add to the app
        router = APIRouter()
        for path in paths:
            router.add_api_route(
                path,
                predict,
                methods=["get"],
                summary=f"{task_name.title()}",
                response_model=resp_model,
                response_model_exclude_unset=True,
                response_model_exclude_none=True,
            )
        self._app.include_router(router)

    def register_taskflow_router(self, task_name):

        # url path to register the model
        paths = [f"/taskflow/{task_name}"]
        print(paths)

        # unique name to create the pydantic model
        unique_name = (hashlib.md5(task_name.encode()).hexdigest())

        # create response model
        resp_model = create_model(
            "V1V1ResponseModel" + unique_name,
            result=(typing.Any, ...),
            __base__=ResponseBase,
        )

        # template predict endpoint function to dynamically serve different models
        def predict(
            request: Request,
            text: str,
        ):
            result = self._app._taskflow_manager.predict(text)
            return {"text": text, 'result': result}

        # register the route and add to the app
        router = APIRouter()
        for path in paths:
            router.add_api_route(
                path,
                predict,
                methods=["get"],
                summary=f"{task_name.title()}",
                response_model=resp_model,
                response_model_exclude_unset=True,
                response_model_exclude_none=True,
            )
        self._app.include_router(router)
