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

import six
import warnings
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.framework import Block
from paddle.fluid import Program
from paddle.fluid.incubate.fleet.parameter_server.ir.program_utils import delete_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_optimize_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_lr_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_sparse_tablenames
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_heter_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_block_joint
from paddle.fluid.incubate.fleet.parameter_server.ir.public import add_vars_by_op_map
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_block_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_op_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_vars_name_in_block
from paddle.fluid.incubate.fleet.parameter_server.ir.public import block_append_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import replace_ops_by_communicate_op
from paddle.fluid.incubate.fleet.parameter_server.ir.program_utils import _get_input_map_from_op, _get_output_map_from_op
from paddle.fluid.incubate.fleet.parameter_server.ir.pserver_pass import add_listen_and_serv_pass


OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "@CLIP"
STEP_COUNTER = "@PS_STEP_COUNTER@"

OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()


def delete_optimizer_pass(program, config):
    def _delete_optimizer_op_and_vars(_program, optimize_ops):
        optimize_vars = []
        optimize_op_role_vars = []
        optimize_need_delete_vars = []

        for op in optimize_ops:
            optimize_vars.extend(op.input_arg_names)
            optimize_op_role_vars.extend(op.attr("op_role_var"))

        optimize_vars = list(set(optimize_vars))
        optimize_op_role_vars = list(set(optimize_op_role_vars))

        for var in optimize_vars:
            if var not in optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        delete_ops(_program.global_block(), optimize_ops)
        for var in need_delete_optimize_vars:
            if _program.global_block().has_var(var):
                _program.global_block()._remove_var(var)

    optimizer_ops = _get_optimize_ops(program)
    lr_ops = _get_lr_ops(program)
    optimizer_ops.extend(lr_ops)
    _delete_optimizer_op_and_vars(program, optimizer_ops)

    return program


def distributed_ops_pass(program, config):
    trainer_id = config.get_role_id()

    def _get_pull_sparse_ops(_program):
        pull_sparse_ops = {}
        op_types = {"lookup_table": "W"}
        for op in _program.global_block().ops:
            if op.type in op_types.keys() \
                    and op.attr('remote_prefetch') is True:
                param_name = op.input(op_types[op.type])[0]
                ops = pull_sparse_ops.get(param_name, [])
                ops.append(op)
                pull_sparse_ops[param_name] = ops
        return pull_sparse_ops

    def _pull_sparse_fuse(_program, pull_sparse_ops):
        for param, ops in pull_sparse_ops.items():
            all_ops = program.global_block().ops
            op_idxs = [all_ops.index(op) for op in ops]
            inputs = [
                program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = program.global_block().vars[ops[0].input("W")[0]]
            padding_idx = ops[0].attr("padding_idx")
            is_distributed = ops[0].attr("is_distributed")

            outputs = [
                program.global_block().vars[op.output("Out")[0]] for op in ops
            ]

            for idx in op_idxs[::-1]:
                program.global_block()._remove_op(idx)

            inputs_idxs = [-1] * len(inputs)
            outputs_idxs = [-1] * len(outputs)

            for idx, op in enumerate(program.global_block().ops):
                for i in range(0, len(op.output_names)):
                    outs = op.output(op.output_names[i])
                    for in_id, in_var in enumerate(inputs):
                        if in_var.name in outs:
                            inputs_idxs[in_id] = idx
                for i in range(0, len(op.input_names)):
                    ins = op.input(op.input_names[i])
                    for out_id, out_var in enumerate(outputs):
                        if out_var.name in ins:
                            outputs_idxs[out_id] = idx

            tables = config.get_var_distributed(w.name, True)

            pserver_endpoints = config.get_ps_endpoints()

            tablenames, eps, sections, = [], [], []
            for table in tables:
                tablenames.append(table[0])
                eps.append(table[1])
                sections.append(table[2])

            if min(outputs_idxs) - max(inputs_idxs) >= 1:
                distributed_idx = max(inputs_idxs) + 1

                program.global_block()._insert_op(
                    index=distributed_idx,
                    type="distributed_lookup_table",
                    inputs={"Ids": inputs,
                            'W': w},
                    outputs={"Outputs": outputs},
                    attrs={
                        "table_names": tablenames,
                        "endpoints": eps,
                        "is_distributed": is_distributed,
                        "pserver_num": len(pserver_endpoints),
                        "padding_idx": padding_idx,
                        "trainer_id": trainer_id
                    })
            else:
                raise ValueError(
                    "something wrong with Fleet, submit a issue is recommended")

    pull_sparse_ops = _get_pull_sparse_ops(program)
    _pull_sparse_fuse(program, pull_sparse_ops)
    return program


def append_send_ops_pass(program, config):
    mode = config.get_distributed_mode()
    trainer_id = config.get_role_id()
    pserver_endpoints = config.get_ps_endpoints()

    def _append_send_op(union_vars, queue):

        if queue == STEP_COUNTER:
            send_input_vars = []
        else:
            send_input_vars = [
                program.global_block().vars[union_var]
                for union_var in union_vars
            ]

        dummy_output = []
        if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            dummy_output = program.global_block().create_var(
                name=framework.generate_control_dev_var_name())

        program.global_block().append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": [queue],
                "merge_add": True,
                "use_send_handler": False,
                "endpoints": pserver_endpoints,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        return dummy_output

    def _append_barrier_op(dummys):
        program.global_block().append_op(
            type="send_barrier",
            inputs={"X": dummys},
            outputs={"Out": []},
            attrs={
                "endpoints": pserver_endpoints,
                "trainer_id": trainer_id,
                "half_async": True,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

    dummys = []

    sends = config.get_trainer_send_context()

    for merged_name, send in sends.items():
        dummys.append(_append_send_op(send.origin_varnames(), merged_name))

    if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
        _append_barrier_op(dummys)

    return program


def init_from_server_pass(program, config):
    fetch_barrier_out = program.global_block().create_var(
        name=framework.generate_control_dev_var_name())

    recv_ctx = config.get_communicator_recv_context(recv_type=1)
    recv_varnames = []

    for name, ctxs in recv_ctx.items():
        recv_varnames.extend(ctxs.origin_varnames())

    program.global_block().append_op(
        type="recv",
        inputs={"X": []},
        outputs={"Out": []},
        attrs={
            "recv_varnames": recv_varnames,
            "trainer_id": config.get_role_id(),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
        })

    program.global_block().append_op(
        type="fetch_barrier",
        inputs={},
        outputs={"Out": fetch_barrier_out},
        attrs={
            "endpoints": config.get_ps_endpoints(),
            "trainer_id": config.get_role_id(),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
        })
    return program


def fake_init_ops_pass(program, config):
    origin_program = config.get_origin_main_program()

    def _get_sparse_table_names():
        dist_varnames = get_sparse_tablenames(origin_program, True)
        sparse_varnames = get_sparse_tablenames(origin_program, False)
        return list(set(dist_varnames + sparse_varnames))

    def _fake_init_sparsetable(sparse_table_names):
        # delete table init op
        for table_name in sparse_table_names:
            table_var = program.global_block().vars[table_name]
            table_param_init_op = []
            for op in program.global_block().ops:
                if table_name in op.output_arg_names:
                    table_param_init_op.append(op)
            init_op_num = len(table_param_init_op)
            if init_op_num != 1:
                raise ValueError("table init op num should be 1, now is " + str(
                    init_op_num))
            table_init_op = table_param_init_op[0]
            program.global_block().append_op(
                type="fake_init",
                inputs={},
                outputs={"Out": table_var},
                attrs={"shape": table_init_op.attr('shape')})
            delete_ops(program.global_block(), table_param_init_op)

    sparse_tables = _get_sparse_table_names()
    _fake_init_sparsetable(sparse_tables)

    return program


def delet_extra_optimizes_pass(program, config):
    optimize_vars = []
    optimize_op_role_vars = []
    optimize_need_delete_vars = []

    origin_program = config.get_origin_main_program()
    for op in _get_optimize_ops(origin_program):
        optimize_vars.extend(op.input_arg_names)
        optimize_op_role_vars.extend(op.attr("op_role_var"))

    optimize_vars = list(set(optimize_vars))
    optimize_op_role_vars = list(set(optimize_op_role_vars))

    for var in optimize_vars:
        if var not in optimize_op_role_vars:
            optimize_need_delete_vars.append(var)
    need_delete_optimize_vars = list(set(optimize_need_delete_vars))

    init_ops = []
    for var in need_delete_optimize_vars:
        param_init_op = []
        for op in program.global_block().ops:
            if var in op.output_arg_names:
                param_init_op.append(op)
        init_ops.extend(param_init_op)
    delete_ops(program.global_block(), init_ops)

    for var in need_delete_optimize_vars:
        if program.global_block().has_var(var):
            program.global_block()._remove_var(var)

    return program


def split_heter_worker_ops_pass(program, config):
    default_deveice = "cpu"
    current_device = "gpu"
    program, heter_ops, _, program_block_ops = find_heter_ops(program, default_deveice)

    if len(heter_ops) == 0:
        return program

    if current_device not in heter_ops:
        raise ValueError(
            "Op which run on device {} not exist.".format(current_device))

    block_vars_detail = find_block_joint(program, program_block_ops)

    heter_program = Program()

    # add heter op
    pre_block_idx = heter_program.num_blocks - 1
    for index, heter_block_ops in heter_ops[current_device].items():
        heter_block = heter_program._create_block(pre_block_idx)
        for _, op in enumerate(heter_block_ops):
            block_append_op(heter_program, program, heter_block, op)

            # add relate variables
            inputs = _get_input_map_from_op(
                program.global_block().vars, op)
            add_vars_by_op_map(inputs, heter_program)

            outputs = _get_output_map_from_op(
                program.global_block().vars, op)
            add_vars_by_op_map(outputs, heter_program)

        # entrance_vars = block_vars_detail[index]["entrance"]
        # exit_vars = block_vars_detail[index]["exit"]

    # attrs = {
    #     "optimize_blocks": optimize_block,
    #     "endpoint": endpoint,
    #     "Fanin": self.trainer_num,
    #     "distributed_mode": DistributedMode.GEO,
    #     "grad_to_block_id": param_to_block_id,
    #     "sparse_grad_to_param": sparse_grad_to_param,
    #     "rpc_get_thread_num": self.server_config._rpc_get_thread_num,
    #     "rpc_send_thread_num": self.server_config._rpc_send_thread_num,
    #     "rpc_prefetch_thread_num":
    #         self.server_config._rpc_prefetch_thread_num
    # }

    # append the listen_and_serv op
    program.global_block().append_op(
        type="listen_and_serv",
        inputs={'X': []},
        outputs={},
        attrs={})

    return program


def split_trainer_ops_pass(program, config):
    default_deveice = "cpu"
    # 复用XPU-Trainer逻辑找到连接点
    origin_program = program.clone()
    origin_program, _, _, program_block_ops = find_heter_ops(origin_program, default_deveice)
    block_vars_detail = find_block_joint(origin_program, program_block_ops)

    block_nums = heter_program.num_blocks
    for block_index in range(1, block_nums):
        current_block = heter_program.block(block_index)
        block_input, block_output = find_block_input_output(
            heter_program, current_block)
        # find entrance & exit
        block_private_vars = list(set(block_input) & set(block_output))
        block_entrance = list(set(block_input)-set(block_private_vars))
        block_exit = list(set(block_output)-set(block_private_vars))

        # delete useless op & add communicate op
        replace_ops_by_communicate_op(origin_program.global_block(),
                                      current_block.ops, origin_program, block_entrance, block_exit, config)
        # delete useless var
        for var in block_private_vars:
            if origin_program.global_block().has_var(var):
                origin_program.global_block()._remove_var(var)

    return origin_program


def delete_startup_useless_ops_var_pass(startup_program, main_program, config):
    # find all op and its var
    vars_in_main_program = get_vars_name_in_block(main_program.global_block())

    block_nums = startup_program.num_blocks
    for block_index in range(1, block_nums):
        current_block = startup_program.block(block_index)
        # delete useless op
        need_delete_op = []
        for op in current_block.ops:
            inputs, outputs = find_op_input_output(
                startup_program, current_block, op)
            inputs += outputs
            # Todo: delete some concat op
            if list(set(inputs) & set(vars_in_main_program)) == None:
                need_delete_op.append(op)
        delete_ops(current_block, need_delete_op)

        # delete useless var
        for var in current_block.vars:
            if var.name not in vars_in_main_program:
                startup_program._remove_var(var.name)

    return startup_program
