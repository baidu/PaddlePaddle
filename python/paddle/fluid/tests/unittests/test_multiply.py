#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.tensor as tensor
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import numpy as np
import unittest


class TestMultiplyAPI(unittest.TestCase):
    def __test_case(self, x_data, y_data, axis=None):
        with program_guard(Program(), Program()):
            x = paddle.nn.data(name='x', shape=x_data.shape, dtype='float32')
            y = paddle.nn.data(name='y', shape=y_data.shape, dtype='float32')
            res = tensor.multiply(x, y, axis=axis)
    
            place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
            exe = fluid.Executor(place)
            outs = exe.run(fluid.default_main_program(), feed={'x':x_data, 'y':y_data}, fetch_list=[res])
            res = outs[0]
            return res
    
    def test_multiply(self):
        # test static computation graph: 1-d array
        x_data = np.array([1, 2, 3], dtype=np.float32)
        y_data = np.array([4, 5, 6], dtype=np.float32)
        res = self.__test_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test static computation graph: 2-d array
        x_data = np.array([[1], [2], [3]], dtype=np.float32)
        y_data = np.array([[4], [5], [6]], dtype=np.float32)
        res = self.__test_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test static computation graph: broadcast
        x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_data = np.array([1, 2, 3], dtype=np.float32)
        res = self.__test_case(x_data, y_data)
        expected = np.array([[1, 4, 9], [4, 10, 18]], dtype=np.float32)
        self.assertTrue(np.allclose(res, expected))

        # test static computation graph: broadcast with axis
        x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        res = self.__test_case(x_data, y_data, axis=0)
        expected = np.array([[1, 2, 3], [8, 10, 12]], dtype=np.float32)
        self.assertTrue(np.allclose(res, expected))

        # test dynamic computation graph
        paddle.enable_imperative()

        # test dynamic computation graph: 1-d array
        x_data = np.array([1, 2, 3], dtype=np.float32)
        y_data = np.array([4, 5, 6], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        res = paddle.multiply(x, y)
        self.assertTrue(np.allclose(res.numpy(), np.multiply(x_data, y_data)))

        # test dynamic computation graph: 2-d array
        x_data = np.array([[1], [2], [3]], dtype=np.float32)
        y_data = np.array([[4], [5], [6]], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        res = paddle.multiply(x, y, axis=1)
        self.assertTrue(np.allclose(res.numpy(), np.multiply(x_data, y_data)))

        # test dynamic computation graph: broadcast
        x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_data = np.array([1, 2, 3], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        res = paddle.multiply(x, y, axis=1)
        expected = np.array([[1, 4, 9], [4, 10, 18]], dtype=np.float32)
        self.assertTrue(np.allclose(res.numpy(), expected))

        # test dynamic computation graph: broadcast with axis
        x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        res = paddle.multiply(x, y, axis=0)
        expected = np.array([[1, 2, 3], [8, 10, 12]], dtype=np.float32)
        self.assertTrue(np.allclose(res.numpy(), expected))

class TestMultiplyError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_imperative()
        
        # dtype can not be int8
        x_data = np.array([1, 2, 3], dtype=np.int8)
        y_data = np.array([1, 2, 3], dtype=np.int8)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        self.assertRaises(fluid.core.EnforceNotMet, paddle.multiply, x, y)

        # inputs must must be broadcastable
        x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_data = np.array([1, 2], dtype=np.float32)
        x = paddle.imperative.to_variable(x_data)
        y = paddle.imperative.to_variable(y_data)
        self.assertRaises(fluid.core.EnforceNotMet, paddle.multiply, x, y)


if __name__ == '__main__':
    unittest.main()
