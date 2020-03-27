#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestLinspaceOpCommonCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([0]).astype(dtype),
            'Stop': np.array([10]).astype(dtype),
            'Num': np.array([11]).astype('int32')
        }

        self.outputs = {'Out': np.arange(0, 11).astype(dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpReverseCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
            'Num': np.array([11]).astype('int32')
        }

        self.outputs = {'Out': np.arange(10, -1, -1).astype(dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpNumOneCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
            'Num': np.array([1]).astype('int32')
        }

        self.outputs = {'Out': np.array(10, dtype=dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceAPI(unittest.TestCase):
    def test_api(self):
        out_1 = fluid.data(name="out_1", shape=[5], dtype="float32")
        out_2 = paddle.tensor.linspace(0, 10, 5, dtype='float32', out=out_1)
        exe = fluid.Executor(place=fluid.CPUPlace())
        ipt = {'out_1': np.random.random([5]).astype('float32')}
        res_1, res_2 = exe.run(fluid.default_main_program(),
                               feed=ipt,
                               fetch_list=[out_1, out_2])
        assert np.array_equal(res_1, res_2)


class TestLinspaceOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # for ci coverage
            # The device of fill_constant must be in 'cpu', 'gpu' or None 
            def test_device_value():
                paddle.tensor.linspace(
                    0, 10, 1, dtype="float32", device='xxxpu')

            self.assertRaises(ValueError, test_device_value)


if __name__ == "__main__":
    unittest.main()
