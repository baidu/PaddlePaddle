# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Lazy imports for heavy dependencies."""

import importlib


def try_import(module_name):
    """Try importing a module, with an informative error message on failure."""
    try:
        mod = importlib.import_module(module_name)
        return mod
    except ImportError:
        err_msg = (
            "Failed importing {name}. This likely means that some paddle modules "
            "requires additional dependencies that have to be "
            "manually installed (usually with `pip install {name}`). ").format(
                name=module_name)
        raise ImportError(err_msg)
