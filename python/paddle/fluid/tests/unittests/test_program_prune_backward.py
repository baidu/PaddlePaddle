#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from test_parallel_executor_mnist import simple_fc_net, fc_with_batchnorm
import seresnext_net
from test_parallel_executor_transformer import transformer, get_feed_data_reader


class TestProgramPruneBackward(unittest.TestCase):
    def program_compare(self, program_a, program_b):
        assert isinstance(
            program_a, fluid.framework.
            Program), "The first argument should be fluid.framework.Program."
        assert isinstance(
            program_b, fluid.framework.
            Program), "The second argument should be fluid.framework Program."

        self.assertEqual(len(program_a.blocks), len(program_b.blocks))
        for idx in range(len(program_a.blocks)):
            block_a = program_a.blocks[idx]
            block_b = program_b.blocks[idx]
            self.assertEqual(len(block_a.ops), len(block_b.ops))
            # self.assertEqual(len(block_a.vars), len(block_b.vars))
            for op_idx in range(len(block_a.ops)):
                self.assertEqual(block_a.ops[op_idx].type,
                                 block_b.ops[op_idx].type)
            for var_key in list(block_a.vars.keys()):
                if block_a.vars[var_key].type == core.VarDesc.VarType.RAW:
                    continue
                self.assertTrue(block_b.has_var(var_key))

    def check_prune_correctness(self, method, feed_dict, optimizer):
        with self.program_scope_guard():
            loss = method(use_feed=False)

            main_program = fluid.default_main_program()
            test_prog_orig = main_program.clone(for_test=True)
            optimizer().minimize(loss)
            test_prog_prune = main_program.clone(for_test=True)

            place = core.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            loss_data_orig, = exe.run(test_prog_orig,
                                      feed=feed_dict,
                                      fetch_list=[loss.name])
            loss_data_prune, = exe.run(test_prog_prune,
                                       feed=feed_dict,
                                       fetch_list=[loss.name])

            self.program_compare(test_prog_orig, test_prog_prune)
            self.assertEqual(loss_data_orig, loss_data_prune)

    def _init_fc_data(self):
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    def test_simple_fc_net(self):
        def optimizer():
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        img, label = self._init_fc_data()
        self.check_prune_correctness(
            method=simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            optimizer=optimizer)

    def test_batchnorm_fc(self):
        def optimizer():
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        img, label = self._init_fc_data()
        self.check_prune_correctness(
            method=fc_with_batchnorm,
            feed_dict={"image": img,
                       "label": label},
            optimizer=optimizer)

    def test_seresnet(self):
        self.check_prune_correctness(
            method=seresnext_net.model,
            feed_dict=seresnext_net.feed_dict(use_cuda=False),
            optimizer=seresnext_net.optimizer)

    def test_transformer(self):
        def optimizer():
            optimizer = fluid.optimizer.Adam(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        # the program argument is used to distinguish Program and CompiledProgram
        feed_dict = get_feed_data_reader().get_next(
            fluid.Executor(core.CPUPlace()), fluid.default_main_program())
        self.check_prune_correctness(
            method=transformer, feed_dict=feed_dict, optimizer=optimizer)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
