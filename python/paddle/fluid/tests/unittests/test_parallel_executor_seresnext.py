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

from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import paddle.fluid.core as core
from parallel_executor_test_base import TestParallelExecutorBase
import unittest
import math
import os
import numpy as np

# FIXME(zcd): If the neural net has dropout_op, the output of ParallelExecutor
# and Executor is different. Because, for ParallelExecutor, the dropout_op of
# the neural net will be copied N copies(N is the number of device). This will
# lead to the random numbers generated by ParallelExecutor and Executor are different.
# So, if we compare the loss of ParallelExecutor and Executor, we should remove the
# dropout_op.
remove_dropout = False

# FIXME(zcd): If the neural net has batch_norm, the output of ParallelExecutor
# and Executor is different.
remove_bn = False


def squeeze_excitation(input, num_channels, reduction_ratio):
    # pool = fluid.layers.pool2d(
    #    input=input, pool_size=0, pool_type='avg', global_pooling=True)
    conv = input
    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)

    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels // reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return conv if remove_bn else fluid.layers.batch_norm(
        input=conv, act=act, momentum=0.1)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    # The number of first 1x1 convolutional channels for each bottleneck build block
    # was halved to reduce the compution cost.
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters * 2,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


batch_size = 12
img_shape = [3, 224, 224]


def SE_ResNeXt50Small(use_feed):

    img = fluid.layers.data(name='image', shape=img_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    conv = conv_bn_layer(
        input=img, num_filters=16, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)
    dropout = pool if remove_dropout else fluid.layers.dropout(
        x=pool, dropout_prob=0.2, seed=1)
    # Classifier layer:
    prediction = fluid.layers.fc(input=dropout, size=1000, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """
    Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * \
                     (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr


def optimizer(learning_rate=0.01):
    optimizer = fluid.optimizer.Momentum(
        learning_rate=cosine_decay(
            learning_rate=learning_rate, step_each_epoch=2, epochs=1),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer


class TestResnet(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)
        global remove_dropout
        global remove_bn
        remove_dropout = False
        remove_bn = False

    def _init_data(self, batch_size=2, random=True):
        np.random.seed(5)
        if random:
            img = np.random.random(
                size=[batch_size] + img_shape).astype(np.float32)
        else:
            img = np.ones(shape=[batch_size] + img_shape, dtype='float32')
        label = [np.random.randint(0, 999) for _ in range(batch_size)]
        label = np.array(label).astype(np.int64).reshape(-1, 1)
        return img, label

    def _compare_reduce_and_allreduce(self,
                                      model,
                                      use_cuda,
                                      iter=20,
                                      delta2=1e-6):
        if use_cuda and not core.is_compiled_with_cuda():
            return

        global remove_bn
        remove_bn = True

        img, label = self._init_data(batch_size=batch_size)
        all_reduce_first_loss, all_reduce_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            use_reduce=False,
            optimizer=optimizer)
        reduce_first_loss, reduce_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            use_reduce=True,
            optimizer=optimizer)

        for loss in zip(all_reduce_first_loss, reduce_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(all_reduce_last_loss, reduce_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

        if not use_cuda:
            return

        all_reduce_first_loss_seq, all_reduce_last_loss_seq = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            use_reduce=False,
            optimizer=optimizer,
            enable_sequential_execution=True)

        reduce_first_loss_seq, reduce_last_loss_seq = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            use_reduce=True,
            optimizer=optimizer,
            enable_sequential_execution=True)

        for loss in zip(all_reduce_first_loss, all_reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(all_reduce_last_loss, all_reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

        for loss in zip(reduce_first_loss, reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(reduce_last_loss, reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

        for loss in zip(all_reduce_first_loss_seq, reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(all_reduce_last_loss_seq, reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

    def _check_resnet_convergence(self,
                                  model,
                                  use_cuda=True,
                                  use_reduce=False,
                                  iter=20,
                                  delta2=1e-6):
        if use_cuda and not core.is_compiled_with_cuda():
            return

        global remove_dropout
        global remove_bn
        remove_dropout = True
        remove_bn = True

        img, label = self._init_data(batch_size=batch_size)
        single_first_loss, single_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            use_reduce=use_reduce,
            optimizer=optimizer,
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            iter=iter,
            batch_size=batch_size,
            use_cuda=use_cuda,
            use_reduce=use_reduce,
            optimizer=optimizer)

        self.assertAlmostEquals(
            np.mean(parallel_first_loss), single_first_loss[0], delta=1e-6)
        self.assertAlmostEquals(
            np.mean(parallel_last_loss), single_last_loss[0], delta=delta2)

    def test_seresnext_with_learning_rate_decay(self):
        if core.is_compiled_with_cuda():
            self._check_resnet_convergence(
                model=SE_ResNeXt50Small, use_cuda=True)
        self._check_resnet_convergence(
            model=SE_ResNeXt50Small, use_cuda=False, iter=2, delta2=1e-3)

    def test_seresnext_with_new_strategy(self):
        self._compare_reduce_and_allreduce(
            model=SE_ResNeXt50Small, use_cuda=True, delta2=1e-2)
        self._compare_reduce_and_allreduce(
            model=SE_ResNeXt50Small, use_cuda=False, iter=5)


if __name__ == '__main__':
    unittest.main()
