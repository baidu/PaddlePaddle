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
"""HDFS Utils."""

import os
import sys
import subprocess
import multiprocessing
from datetime import datetime

import re
import copy
import errno
import time
import logging
from . import fs
from .fs import FS
import paddle.fluid as fluid
import functools

from pathlib import PurePosixPath
import shutil

__all__ = ["HDFSClient"]


class ExecuteError(Exception):
    pass


class FSFileExistsError(Exception):
    pass


class FSFileiNotExistsError(Exception):
    pass


def _handle_errors(f):
    def handler(*args, **kwargs):
        start = time.time()
        while True:
            try:
                return f(*args, **kwargs)
            except ExecuteError as e:
                o = args[0]
                time_out = float(o._time_out) / 1000.0
                inter = float(o._sleep_inter) / 1000.0
                if time.time() - start >= time_out:
                    break
                time.sleep(inter)

    return functools.wraps(f)(handler)


class HDFSClient(FS):
    def __init__(
            self,
            hadoop_home,
            configs,
            time_out=5 * 60 * 1000,  #ms
            sleep_inter=1000):  #ms
        self.pre_commands = []
        hadoop_bin = '%s/bin/hadoop' % hadoop_home
        self.pre_commands.append(hadoop_bin)
        dfs = 'fs'
        self.pre_commands.append(dfs)

        if configs:
            for k, v in configs.iteritems():
                config_command = '-D%s=%s' % (k, v)

        self._time_out = time_out
        self._sleep_inter = sleep_inter
        self._base_cmd = " ".join(self.pre_commands)

    def _run_cmd(self, cmd, redirect_stderr=False):
        ret, output = fluid.core.shell_execute_cmd(cmd, 0, 0, redirect_stderr)
        print("run_cmd", ret, output)
        return int(ret), output.splitlines()

    def list_dirs(self, fs_path):
        if not self.is_exist(fs_path):
            return []

        dirs, _ = self._ls_dir(fs_path)
        return dirs

    @_handle_errors
    def ls_dir(self, fs_path):
        """	
        list directory under fs_path, and only give the pure name, not include the fs_path	
        """
        if not self.is_exist(fs_path):
            return [], []

        cmd = "{} -ls {}".format(self._base_cmd, fs_path)
        print("ls_dir cmd:", cmd)
        ret, lines = self._run_cmd(cmd)

        if ret != 0:
            raise ExecuteError

        dirs = []
        files = []
        for line in lines:
            arr = line.split()
            if len(arr) != 8:
                continue

            if fs_path not in arr[7]:
                continue

            p = PurePosixPath(arr[7])
            if arr[0][0] == 'd':
                dirs.append(p.name)
            else:
                files.append(p.name)

        return dirs, files

    @_handle_errors
    def is_dir(self, fs_path):
        if not self.is_exist(fs_path):
            return False

        cmd = "{} -test -d {} ".format(self._base_cmd, fs_path)
        ret, lines = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

        return True if ret == 0 else False

    def is_file(self, fs_path):
        if not self.is_exist(fs_path):
            return False

        return not self.is_dir(fs_path)

    @_handle_errors
    def is_exist(self, fs_path):
        cmd = "{} -ls {} ".format(self._base_cmd, fs_path)
        ret, out = self._run_cmd(cmd, redirect_stderr=True)
        print("is exist return:", cmd, ret, out)
        if ret != 0:
            for l in out:
                if "No such file or directory" in l:
                    print(l)
                    return False
            raise ExecuteError

        return True

    @_handle_errors
    def upload(self, local_path, fs_path):
        if self.is_exist(fs_path):
            raise FSFileExistsError

        if not self._local.is_exist(local_path):
            raise FSFileNotExistsError

        cmd = "{} -put {} {}".format(self._base_cmd, local_path, fs_path)
        ret, lines = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    @_handle_errors
    def download(self, fs_path, local_path):
        if self.is_exist(local_path):
            raise FSFileExistsError

        if not self.is_exist(fs_path):
            raise FSFileNotExistsError

        cmd = "{} -get {} {}".format(self._base_cmd, fs_path, local_path)
        ret, lines = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    @_handle_errors
    def mkdirs(self, fs_path):
        if self.is_exist(fs_path):
            return

        cmd = "{} -mkdir {}".format(self._base_cmd, fs_path)
        ret, lines = self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    @_handle_errors
    def mv(self, fs_src_path, fs_dst_path):
        if self.is_exist(fs_dst_path):
            raise FSFileExistsError

        if not self.is_exist(fs_src_path):
            raise FSFileNotExistsError

        cmd = "{} -mv {} {}".format(self._base_cmd, fs_src_path, fs_dst_path)
        self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    @_handle_errors
    def _rmr(self, fs_path):
        if not self.is_exist(fs_path):
            return

        cmd = "{} -rmr {}".format(self._base_cmd, fs_path)
        self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    @_handle_errors
    def _rm(self, fs_path):
        if not self.is_exist(fs_path):
            return

        cmd = "{} -rm {}".format(self._base_cmd, fs_path)
        self._run_cmd(cmd)
        if ret != 0:
            raise ExecuteError

    def delete(self, fs_path):
        if not self.is_exist(fs_path):
            return

        is_dir = self.is_dir(fs_path)
        if is_dir:
            return self._rmr(fs_path)

        return self._rm(fs_path)

    def need_upload_download(self):
        return True
