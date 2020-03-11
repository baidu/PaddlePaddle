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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_graph

SEED = 2020
np.random.seed(SEED)


def basic_test(x):
    x = fluid.dygraph.to_variable(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
    else:
        a.append(
            fluid.layers.fill_constant(
                shape=[1, 2], value=9, dtype="int64"))
    return a


test_funcs = [basic_test]


class TestArray(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random((3)).astype('int32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            if isinstance(res, (list, tuple)):
                res = res[0]
            return res.numpy()

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            tensor_array = dygraph_to_static_graph(self.dygraph_func)(
                self.input)
            static_out = fluid.layers.array_read(
                tensor_array,
                i=fluid.layers.fill_constant(
                    shape=[1], value=0, dtype='int64'))
        exe = fluid.Executor(self.place)
        static_res = exe.run(main_program, fetch_list=static_out)

        return static_res[0]

    def test_transformed_static_result(self):
        for func in test_funcs:
            self.dygraph_func = func
            static_res = self.run_static_mode()
            dygraph_res = self.run_dygraph_mode()
            self.assertTrue(
                np.allclose(dygraph_res, static_res),
                msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                                 static_res))


if __name__ == '__main__':
    unittest.main()
