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

from __future__ import print_function


def delete_optimizer_pass(program):
    return program


def append_send_ops_pass(program):
    return program


def distributed_ops_pass(program):
    return program


def fake_init_ops_pass(program):
    return program


def get_communicator_context(program):
    send_context = []
    recv_context = []
    return send_context, recv_context
