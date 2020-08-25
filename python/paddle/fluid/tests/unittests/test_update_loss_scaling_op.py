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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid


class TestUpdateLossScalingOp(OpTest):
    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        found_inf = np.array([False], dtype=np.bool)
        x = np.random.random((1024, 1024)).astype(self.dtype)

        self.inputs = {
            'X': [('x0', x)],
            'FoundInfinite': found_inf,
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps
        }

        self.outputs = {
            'Out': [('out0', np.zeros_like(x))],
            'LossScaling': self.prev_loss_scaling * self.incr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps
        }

    def init(self):
        self.incr_ratio = 2.0
        self.decr_ratio = 0.8
        self.dtype = np.float32
        self.prev_loss_scaling = np.array([2048]).astype(self.dtype)
        self.num_good_steps = np.array([999], dtype=np.int32)
        self.num_bad_steps = np.array([1], dtype=np.int32)
        self.zero_steps = np.array([0], dtype=np.int32)
        self.attrs = {
            'incr_every_n_steps': 1000,
            'decr_every_n_nan_or_inf': 2,
            'incr_ratio': self.incr_ratio,
            'decr_ratio': self.decr_ratio,
        }

    def test_check_output(self):
        self.check_output(no_check_set=['Out'])


class TestUpdateLossScalingOpBad(TestUpdateLossScalingOp):
    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        found_inf = np.array([True], dtype=np.bool)
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.inf

        self.inputs = {
            'X': [('x0', x)],
            'FoundInfinite': found_inf,
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps
        }

        self.outputs = {
            'Out': [('out0', np.zeros_like(x))],
            'LossScaling': self.prev_loss_scaling * self.decr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
