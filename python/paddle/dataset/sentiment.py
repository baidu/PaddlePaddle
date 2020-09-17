# /usr/bin/env python
# -*- coding:utf-8 -*-

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
The script fetch and preprocess movie_reviews data set that provided by NLTK

TODO(yuyang18): Complete dataset.
"""

from __future__ import print_function

import six
import collections
import functools
from itertools import chain

import os
import zipfile
from functools import cmp_to_key

import paddle.dataset.common

URL = "https://corpora.bj.bcebos.com/movie_reviews%2Fmovie_reviews.zip"
MD5 = '155de2b77c6834dd8eea7cbe88e93acb'

__all__ = ['train', 'test', 'get_word_dict']
NUM_TRAINING_INSTANCES = 1600
NUM_TOTAL_INSTANCES = 2000


def check_nltk_intall(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import sys
            import nltk
            from nltk.corpus import movie_reviews
            module = sys.modules[__name__]
            setattr(module, "nltk", nltk)
            setattr(module, "movie_reviews", movie_reviews)
        except ImportError as e:
            raise ImportError(e.message + "\n Please install nltk.")
        return func(*args, **kwargs)

    return wrapper


@check_nltk_intall
def download_data_if_not_yet():
    """
    Download the data set, if the data set is not download.
    """
    try:
        # download and extract movie_reviews.zip
        paddle.dataset.common.download(
            URL, 'corpora', md5sum=MD5, save_name='movie_reviews.zip')
        path = os.path.join(paddle.dataset.common.DATA_HOME, 'corpora')
        filename = os.path.join(path, 'movie_reviews.zip')
        zip_file = zipfile.ZipFile(filename)
        zip_file.extractall(path)
        zip_file.close()
        # make sure that nltk can find the data
        if paddle.dataset.common.DATA_HOME not in nltk.data.path:
            nltk.data.path.append(paddle.dataset.common.DATA_HOME)
        movie_reviews.categories()
    except LookupError:
        print("Downloading movie_reviews data set, please wait.....")
        nltk.download(
            'movie_reviews', download_dir=paddle.dataset.common.DATA_HOME)
        print("Download data set success.....")
        print("Path is " + nltk.data.find('corpora/movie_reviews').path)


@check_nltk_intall
def get_word_dict():
    """
    Sorted the words by the frequency of words which occur in sample
    :return:
        words_freq_sorted
    """
    words_freq_sorted = list()
    word_freq_dict = collections.defaultdict(int)
    download_data_if_not_yet()

    for category in movie_reviews.categories():
        for field in movie_reviews.fileids(category):
            for words in movie_reviews.words(field):
                word_freq_dict[words] += 1
    words_sort_list = list(six.iteritems(word_freq_dict))
    words_sort_list.sort(key=cmp_to_key(lambda a, b: b[1] - a[1]))
    for index, word in enumerate(words_sort_list):
        words_freq_sorted.append((word[0], index))
    return words_freq_sorted


@check_nltk_intall
def sort_files():
    """
    Sorted the sample for cross reading the sample
    :return:
        files_list
    """
    files_list = list()
    neg_file_list = movie_reviews.fileids('neg')
    pos_file_list = movie_reviews.fileids('pos')
    files_list = list(
        chain.from_iterable(list(zip(neg_file_list, pos_file_list))))
    return files_list


@check_nltk_intall
def load_sentiment_data():
    """
    Load the data set
    :return:
        data_set
    """
    data_set = list()
    download_data_if_not_yet()
    words_ids = dict(get_word_dict())
    for sample_file in sort_files():
        words_list = list()
        category = 0 if 'neg' in sample_file else 1
        for word in movie_reviews.words(sample_file):
            words_list.append(words_ids[word.lower()])
        data_set.append((words_list, category))
    return data_set


@check_nltk_intall
def reader_creator(data):
    """
    Reader creator, generate an iterator for data set
    :param data:
        train data set or test data set
    """
    for each in data:
        yield each[0], each[1]


@check_nltk_intall
def train():
    """
    Default training set reader creator
    """
    data_set = load_sentiment_data()
    return reader_creator(data_set[0:NUM_TRAINING_INSTANCES])


@check_nltk_intall
def test():
    """
    Default test set reader creator
    """
    data_set = load_sentiment_data()
    return reader_creator(data_set[NUM_TRAINING_INSTANCES:])


@check_nltk_intall
def fetch():
    nltk.download('movie_reviews', download_dir=paddle.dataset.common.DATA_HOME)
