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
import paddle.fluid as fluid
from paddle.fluid.framework import convert_np_dtype_to_dtype_, Program, program_guard
import paddle.fluid.core as core
import numpy as np
import copy
import unittest
import sys
sys.path.append("../")
from op_test import OpTest


class TestSequenceFirstStepOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.randint(1, 5, [4]).astype("int64")

            def test_Variable():
                # the input must be Variable
                fluid.layers.sequence_first_step(input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_input_level():
                # the elements of input must less than maxlen
                y = fluid.layers.data(
                    name='y', shape=[1], dtype='float32', lod_level=3)
                fluid.layers.sequence_first_step(y)

            self.assertRaises(TypeError, test_input_level)

            def test_input_dtype():
                # the dtype of input must be int64
                fluid.layers.sequence_first_step(input_data)

            self.assertRaises(TypeError, test_input_dtype)


if __name__ == '__main__':
    unittest.main()
