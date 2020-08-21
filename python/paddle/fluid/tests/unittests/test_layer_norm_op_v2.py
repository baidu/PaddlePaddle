# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from op_test import OpTest, _set_use_system_allocator
from paddle.fluid.framework import grad_var_name
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle


class TestDygraphLayerNormv2(unittest.TestCase):
    def test_functional_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("batch_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            with fluid.dygraph.guard(p):
                x_data = np.random.random(size=(2, 1, 2, 3)).astype('float32')
                x = fluid.dygraph.to_variable(x_data)
                layer_norm_out = paddle.nn.functional.layer_norm(x, x.shape[1:])

    def test_functional_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("batch_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            with program_guard(Program(), Program()):
                x_data = np.random.random(size=(2, 1, 2, 3)).astype('float32')
                x = fluid.data(name='x', shape=x_data.shape, dtype=x_data.dtype)
                layer_norm_out = paddle.nn.functional.layer_norm(x, [1, 2, 3])
                exe.run(fluid.default_startup_program())
                r = exe.run(feed={'x': x_data, },
                            fetch_list=[layer_norm_out])[0]

    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    ln = fluid.dygraph.LayerNorm(shape[1:])
                    y = ln(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            self.assertTrue(np.allclose(y1, y2))

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ln = fluid.dygraph.LayerNorm(shape[1:])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = ln(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = ln(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            self.assertTrue(np.allclose(y1, y2))


if __name__ == '__main__':
    unittest.main()
