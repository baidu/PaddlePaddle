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
import paddle.fluid.core as core


def numpy_topk(x, k=1, axis=-1, largest=True):
    if axis < 0:
        axis = len(x.shape) + axis
    if largest:
        indices = np.argsort(-x, axis=axis)
    else:
        indices = np.argsort(x, axis=axis)
    indices = indices.take(indices=range(0, k), axis=axis)
    result = np.take_along_axis(x, indices, axis=axis)
    return result, indices


class TestTopkOp(OpTest):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float64
        self.input_data = np.random.rand(10, 20)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out')


class TestTopOp1(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 0
        self.largest = True


class TestTopOp2(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 0
        self.largest = False


class TestTopOp3(TestTopkOp):
    def init_args(self):
        self.k = 4
        self.axis = 0
        self.largest = False


class TestTopOp4(TestTopkOp):
    def init_args(self):
        self.k = 4
        self.axis = 0
        self.largest = False


class TestTopkOp5(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float64
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp6(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float64
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopKAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.input_data = np.random.rand(6, 7, 8)

    def run_static(self, place):
        pass

    def run_dygraph(self, place):
        paddle.disable_static(place)
        input_tensor = paddle.to_variable(self.input_data)
        # test case for basic test case 1
        paddle_result = paddle.topk(input_tensor, k=2)
        numpy_result = numpy_topk(self.input_data, k=2)
        self.assertTrue(np.allclose(paddle_result[0].numpy(), numpy_result[0]))
        self.assertTrue(np.allclose(paddle_result[1].numpy(), numpy_result[1]))
        # test case for basic test case 2 with axis
        paddle_result = paddle.topk(input_tensor, k=2, axis=1)
        numpy_result = numpy_topk(self.input_data, k=2, axis=1)
        self.assertTrue(np.allclose(paddle_result[0].numpy(), numpy_result[0]))
        self.assertTrue(np.allclose(paddle_result[1].numpy(), numpy_result[1]))
        # test case for basic test case 3 with tensor K
        k_tensor = paddle.to_variable(np.array([2]))
        paddle_result = paddle.topk(input_tensor, k=k_tensor, axis=1)
        numpy_result = numpy_topk(self.input_data, k=2, axis=1)
        self.assertTrue(np.allclose(paddle_result[0].numpy(), numpy_result[0]))
        self.assertTrue(np.allclose(paddle_result[1].numpy(), numpy_result[1]))
        # test case for basic test case 4 with tensor largest
        k_tensor = paddle.to_variable(np.array([2]))
        paddle_result = paddle.topk(input_tensor, k=2, axis=1, largest=False)
        numpy_result = numpy_topk(self.input_data, k=2, axis=1, largest=False)
        self.assertTrue(np.allclose(paddle_result[0].numpy(), numpy_result[0]))
        self.assertTrue(np.allclose(paddle_result[1].numpy(), numpy_result[1]))
        # test case for basic test case 5 with axis -1
        k_tensor = paddle.to_variable(np.array([2]))
        paddle_result = paddle.topk(input_tensor, k=2, axis=-1, largest=False)
        numpy_result = numpy_topk(self.input_data, k=2, axis=-1, largest=False)
        self.assertTrue(np.allclose(paddle_result[0].numpy(), numpy_result[0]))
        self.assertTrue(np.allclose(paddle_result[1].numpy(), numpy_result[1]))
        # test case for basic test case 6 for the partial sort 
        fixed_numpy = np.random.rand(2, 100)
        paddle_result = paddle.topk(input_tensor, k=1, axis=-1)
        numpy_result = numpy_topk(self.input_data, k=1, axis=-1)
        self.assertTrue(np.allclose(paddle_result[0].numpy(), numpy_result[0]))
        self.assertTrue(np.allclose(paddle_result[1].numpy(), numpy_result[1]))
        # test case for basic test case 7 for the unsorted 
        paddle_result = paddle.topk(input_tensor, k=2, axis=1, sorted=False)
        sort_paddle = numpy_topk(
            np.array(paddle_result[0].numpy()), axis=1, k=2)
        numpy_result = numpy_topk(self.input_data, k=2, axis=1)
        #print("sort:{}".format(np.sort(paddle_result[0].numpy(), axis=1)[::-1]))
        self.assertTrue(np.allclose(sort_paddle[0], numpy_result[0]))

    def test_cases(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.run_dygraph(place)


if __name__ == "__main__":
    unittest.main()
