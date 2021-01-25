# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
from distutils.sysconfig import get_python_lib
from setuptools import setup
from cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import Extension

file_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_path = get_python_lib()
# Note(Aurelius84): We use `add_test` in cmake to config how to run unittest in CI. 
# `PYTHONPATH` will be set as `build/python/paddle` that will make no way to find
# paddle include directory. Because the following path is generated after insalling
# PaddlePaddle whl. So here we specific `include_dirs` to avoid errors in CI.
paddle_includes = [
    os.path.join(site_packages_path, 'paddle/include'),
    os.path.join(site_packages_path, 'paddle/include/thirdparty')
]

setup(
    name='relu_op_shared',
    ext_modules=[
        CUDAExtension(
            name='librelu2_op_from_setup',
            sources=['relu_op.cc', 'relu_op.cu'],
            include_dirs=paddle_includes,
            output_dir=file_dir)
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
