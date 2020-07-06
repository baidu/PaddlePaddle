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

from .wrapped_decorator import signature_safe_contextmanager
import sys
from paddle.fluid import core
from paddle.fluid import framework
import logging
from contextlib import contextmanager
from . import unique_name
import paddle.fluid as fluid
from fluid.incubate.fleet.utils.hdfs import HDFSClient
import hashlib


class AutoCheckpointChecker(object):
    def __init__(self):
        self._run_env = None
        self._plat_form = None
        self._job_id = None
        self._hdfs_home = None
        self._hdfs_name = None
        self._hdfs_ugi = None
        self._hdfs_checkpoint_path = None
        self._trainer_id = None
        try:
            self._run_env = os.environ("PADDLE_RUNNING_ENV")
            self._plat_form = os.environ("PADDLE_RUNNING_PLATFORM")
            self._job_id = os.environ("PADDLE_JOB_ID")
            self._hdfs_home = os.environ("PADDLE_EDL_HDFS_HOME")
            self._hdfs_name = os.environ("PADDLE_EDL_HDFS_NAME")
            self._hdfs_ugi = os.environ("PADDLE_EDL_HDFS_UGI")
            self._hdfs_checkpoint_path = os.environ(
                "PADDLE_EDL_HDFS_CHECKPOINT_PATH")
            self._trainer_id = int(os.getenv("PADDLE_EDL_TRAINER_ID"))
        except Exception as e:
            #logging.warning("auto checkpoint must run under PADDLE_RUNNING_ENV,PADDLE_RUNNING_PLATFORM,PADDLE_JOB_ID:{}".format(e))
            return

    def get_job_checkpoint_path(self):
        return "{}/{}".format(self._hdfs_checkpoint_path, self._job_id)

    def valid(self):
        return self._run_env is not None and \
            self._plat_form is not None and \
            self._job_id is not None and \
            self._hdfs_home is not None and \
            self._hdfs_name is not None and \
            self._hdfs_ugi is not None and \
            self._hdfs_checkpoint_path is not None and \
            self._trainer_id is not None


class EpochStatus(object):
    def __init__(self):
        self._epoch_no = None
        self._hash_key = None


class TrainStatus(object):
    def __init__(self, epoch_no=-1):
        # completed epoch
        self._epoch_no = epoch_no

    def next(self):
        return self._epoch_no + 1

    def __eq__(self, t):
        return self._epoch_no == t._epoch_no

    def __ne__(self, t):
        return not self == t


g_train_epoch_range = None
g_checker = AutoCheckpointChecker()


class TrainEpochRange(object):
    def __init__(self, max_epoch_num, name):
        self._max_epoch_num = max_epoch_num
        self._epoch_no = -1
        self._name = name

        self._checker = g_checker
        if not self._checker.valid():
            return

        config = {
            "fs.default.name": self._hdfs_name,
            "hadoop.job.ugi": self.hdfs_ugi
        }

        self._hdfs = HDFSClient(self._hadoop_home, config)
        self._job_cp_path = "{}/{}".format(
            self._checker.get_job_checkpoint_path, name)

        self._cper = Checkpointer(self._hdfs)
        self._cp_no = self._cper.get_last_checkpoint_no(self._job_cp_path)
        if self._cp_no < 0:
            return

        train_status = self.load_train_status(self._job_cp_path)
        self._epoch_no = train_status["epoch_no"]

    def __enter__(self):
        print('enter')
        return self

    def __exit__(self):
        print('exit')

    def get(self):
        if self._max_epoch_num < 0:
            self._max_epoch_num = sys.maxint - 1

        assert self._epoch_no >= -1, "self._epoch_no:{} must >=0".format(
            self._epoch_no)
        for i in range(self._epoch_no + 1, self._max_epoch_num + 1):
            yield i


def _get_train_epoch_range_obj():
    return g_train_epoch_range


def _can_auto_checkpoint():
    return g_checker.valid() and g_train_epoch_range is not None


def train_epoch_range(max_epoch_num):
    g_train_epoch_range = TrainEpochRange(
        max_epoch_num, unique_name.generate("train_epoch_range"))
    for i in t.get():
        yield i
    g_train_epoch_range = None


def _initial_ids(exe, program, io_key):
    if program._auto_checkpoint_name is None:
        program._auto_checkpoint_name = _generate_program_name()

    if exe._auto_checkpoint_name is None:
        exe._auto_checkpoint_name = _generate_executor_name()

    k = "%s_%s_%s".format(exe._auto_checkpoint_name,
                          program._auto_checkpoint_name, io_key)
    h = _get_hash(key)
    logging.info("init auto checkpoint h:{} from key:{}".format(h, k))

    if k not in exe._auto_checkpoint_epoch_status:
        exe._auto_checkpoint_epoch_status[k] = EpochStatus()
        exe._auto_checkpoint_epoch_status._hash_key = h


def _get_hash(key):
    k = key
    if sys.version_info[0] >= 3:
        k = key.encode('utf-8')

    return hashlib.md5(k).hexdigest()


def _try_to_load_checkpoint(exe, program, io_key):
    pass


def _try_to_save_checkpoint(exe, program, io_key):
    pass


def _generate_program_name():
    return unique_name.generate(g_checker._job_id + "_program_")


def _generate_executor_name():
    return unique_name.generate(g_checker._job_id + "_executor_")
