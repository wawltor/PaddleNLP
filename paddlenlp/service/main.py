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

import sys
import atexit
import argparse
import os
from multiprocessing import Process

import uvicorn
from uvicorn.config import LOGGING_CONFIG
from uvicorn.main import LEVEL_CHOICES

from ..utils.log import logger


def start_backend(app, **kwargs):
    uvicorn.run(app, **kwargs)


def parse_args(command_args):
    """
    The argument parser, get the argument of the server.
        args (List[str]): The list of input argument.
    """
    if len(command_args) == 0:
        logger.error(
            'You must set the application name, for example ppnlp-server app !')
        sys.exit(0)
    app = command_args[0]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",
                        default="127.0.0.1",
                        type=str,
                        help="Bind socket to this host.")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Bind socket to this port.")
    parser.add_argument(
        "--debug",
        default=False,
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--workers",
        default=None,
        type=int,
        help=
        "Number of worker processes. Defaults to the $WEB_CONCURRENCY environment"
        " variable if available, or 1. Not valid with --reload.")
    parser.add_argument("--log-level",
                        type=LEVEL_CHOICES,
                        default=None,
                        help="Log level. [default: info]")
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=None,
        help=
        "Maximum number of concurrent connections or tasks to allow, before issuing"
        " HTTP 503 responses.")
    parser.add_argument(
        "--limit-max-requests",
        type=int,
        default=None,
        help=
        "Maximum number of requests to service before terminating the process.")
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=15,
        help=
        "Close Keep-Alive connections if no new data is received within this timeout."
    )
    parser.add_argument("--reload",
                        default=False,
                        help="Enable backend auto-reload.")
    args = parser.parse_args(command_args[1:])
    return app, args


def main():
    args = sys.argv
    app, args = parse_args(args[1:])
    print(args)

    # Flags of uvicorn
    backend_kwargs = {
        "host": args.host,
        "port": args.port,
        "log_config": LOGGING_CONFIG,
        "log_level": args.log_level,
        "debug": True,  #args.debug,
        "workers": args.workers,
        "limit_concurrency": args.limit_concurrency,
        "limit_max_requests": args.limit_max_requests,
        "timeout_keep_alive": args.timeout_keep_alive,
        "reload": args.reload
    }

    # Start backend
    start_backend(app, **backend_kwargs)


if __name__ == "__main__":
    main()  # pragma: no cover
