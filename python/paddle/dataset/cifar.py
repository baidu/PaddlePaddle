# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
CIFAR dataset.

This module will download dataset from
https://www.cs.toronto.edu/~kriz/cifar.html and parse train/test set into
paddle reader creators.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test
images.

The CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes
containing 600 images each. There are 500 training images and 100 testing
images per class.

"""

from __future__ import print_function

import itertools
import numpy
import paddle.dataset.common
import tarfile
import six
from six.moves import cPickle as pickle

__all__ = ['train100', 'test100', 'train10', 'test10', 'convert']

URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


def reader_creator(filename, sub_name, cycle=False):
    def read_batch(batch):
        data = batch[six.b('data')]
        labels = batch.get(
            six.b('labels'), batch.get(six.b('fine_labels'), None))
        assert labels is not None
        for sample, label in six.moves.zip(data, labels):
            yield (sample / 255.0).astype(numpy.float32), int(label)

    def reader():
        with tarfile.open(filename, mode='r') as f:
            names = (each_item.name for each_item in f
                     if sub_name in each_item.name)

            while True:
                for name in names:
                    if six.PY2:
                        batch = pickle.load(f.extractfile(name))
                    else:
                        batch = pickle.load(
                            f.extractfile(name), encoding='bytes')
                    for item in read_batch(batch):
                        yield item
                if not cycle:
                    break

    return reader


def train100():
    """
    CIFAR-100 training set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 99].

    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(CIFAR100_URL, 'cifar', CIFAR100_MD5),
        'train')


def test100():
    """
    CIFAR-100 test set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(CIFAR100_URL, 'cifar', CIFAR100_MD5),
        'test')


def train10(cycle=False):
    """
    CIFAR-10 training set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        'data_batch',
        cycle=cycle)


def test10(cycle=False):
    """
    CIFAR-10 test set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        'test_batch',
        cycle=cycle)


def fetch():
    paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5)
    paddle.dataset.common.download(CIFAR100_URL, 'cifar', CIFAR100_MD5)


def convert(path):
    """
    Converts dataset to recordio format
    """
    paddle.dataset.common.convert(path, train100(), 1000, "cifar_train100")
    paddle.dataset.common.convert(path, test100(), 1000, "cifar_test100")
    paddle.dataset.common.convert(path, train10(), 1000, "cifar_train10")
    paddle.dataset.common.convert(path, test10(), 1000, "cifar_test10")
