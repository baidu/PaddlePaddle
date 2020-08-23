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

import tarfile
import numpy as np
import six
from six.moves import cPickle as pickle

from paddle.io import Dataset
from .utils import _check_exists_and_download

__all__ = ['Cifar10', 'Cifar100']

URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

MODE_FLAG_MAP = {
    'train10': 'data_batch',
    'test10': 'test_batch',
    'train100': 'train',
    'test100': 'test'
}


class Cifar10(Dataset):
    """
    Implementation of `Cifar-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 10 categories.

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'test' mode. Default 'train'.
        transform(callable): transform to perform on image, None for on transform.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of cifar-10 dataset

    Examples:

        .. code-block:: python

	    import paddle
	    from paddle.incubate.hapi.datasets import Cifar10
	    from paddle.incubate.hapi.vision.transforms import Normalize

	    class SimpleNet(paddle.nn.Layer):
		def __init__(self):
		    super(SimpleNet, self).__init__()
		    self.fc = paddle.nn.Linear(3072, 10, act='softmax')

		def forward(self, image, label):
		    image = paddle.reshape(image, (3, -1))
		    return self.fc(image), label

	    paddle.disable_static()

	    normalize = Normalize(mean=[0.5, 0.5, 0.5],
				std=[0.5, 0.5, 0.5])
	    cifar10 = Cifar10(mode='train', transform=normalize)

	    for i in range(10):
		image, label = cifar10[i]
		image = paddle.to_tensor(image)
		label = paddle.to_tensor(label)

		model = SimpleNet()
		image, label = model(image, label)
		print(image.numpy().shape, label.numpy().shape)

    """

    def __init__(self,
                 data_file=None,
                 mode='train',
                 transform=None,
                 download=True):
        assert mode.lower() in ['train', 'test', 'train', 'test'], \
            "mode should be 'train10', 'test10', 'train100' or 'test100', but got {}".format(mode)
        self.mode = mode.lower()

        self._init_url_md5_flag()

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, self.data_url, self.data_md5, 'cifar', download)

        self.transform = transform

        # read dataset into memory
        self._load_data()

    def _init_url_md5_flag(self):
        self.data_url = CIFAR10_URL
        self.data_md5 = CIFAR10_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '10']

    def _load_data(self):
        self.data = []
        with tarfile.open(self.data_file, mode='r') as f:
            names = (each_item.name for each_item in f
                     if self.flag in each_item.name)

            for name in names:
                if six.PY2:
                    batch = pickle.load(f.extractfile(name))
                else:
                    batch = pickle.load(f.extractfile(name), encoding='bytes')

                data = batch[six.b('data')]
                labels = batch.get(
                    six.b('labels'), batch.get(six.b('fine_labels'), None))
                assert labels is not None
                for sample, label in six.moves.zip(data, labels):
                    self.data.append((sample, label))

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)


class Cifar100(Cifar10):
    """
    Implementation of `Cifar-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 100 categories.

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'test' mode. Default 'train'.
        transform(callable): transform to perform on image, None for on transform.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of cifar-100 dataset

    Examples:

        .. code-block:: python

	    import paddle
	    from paddle.incubate.hapi.datasets import Cifar100
	    from paddle.incubate.hapi.vision.transforms import Normalize

	    class SimpleNet(paddle.nn.Layer):
		def __init__(self):
		    super(SimpleNet, self).__init__()
		    self.fc = paddle.nn.Linear(3072, 100, act='softmax')

		def forward(self, image, label):
		    image = paddle.reshape(image, (3, -1))
		    return self.fc(image), label

	    paddle.disable_static()

	    normalize = Normalize(mean=[0.5, 0.5, 0.5],
				std=[0.5, 0.5, 0.5])
	    cifar100 = Cifar100(mode='train', transform=normalize)

	    for i in range(10):
		image, label = cifar100[i]
		image = paddle.to_tensor(image)
		label = paddle.to_tensor(label)

		model = SimpleNet()
		image, label = model(image, label)
		print(image.numpy().shape, label.numpy().shape)

    """

    def __init__(self,
                 data_file=None,
                 mode='train',
                 transform=None,
                 download=True):
        super(Cifar100, self).__init__(data_file, mode, transform, download)

    def _init_url_md5_flag(self):
        self.data_url = CIFAR100_URL
        self.data_md5 = CIFAR100_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '100']
