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
import paddle
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestWhereZklOp(OpTest):
    def setUp(self):
        self.op_type = "where_zkl"
        self.dtype = np.float64
        self.init_dtype_type()

        X = np.random.random((200, 1)).astype(self.dtype)
        Y = np.random.random((200, 1)).astype(self.dtype)
        Condition = X > 1
        out = np.where(Condition, X, Y)

        self.inputs = {'Condition': Condition, 'X': X, 'Y': Y}
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


if __name__ == '__main__':
    unittest.main()
