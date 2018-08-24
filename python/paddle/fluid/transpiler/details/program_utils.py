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

import six

from paddle.fluid import core
import paddle


def delete_ops(block, ops):
    try:
        start = list(block.ops).index(ops[0])
        end = list(block.ops).index(ops[-1])
        [block._remove_op(start) for _ in six.moves.range(end - start + 1)]
    except Exception as e:
        raise e
    block.program._sync_with_cpp()


def find_op_by_input_arg(block, arg_name):
    for index, op in enumerate(block.ops):
        if arg_name in op.input_arg_names:
            return index
    return -1


def find_op_by_output_arg(block, arg_name):
    for index, op in enumerate(block.ops):
        if arg_name in op.output_arg_names:
            return index
    return -1


def get_indent_space(indent, space_num=4):
    ret = ""
    for i in range(0, indent * space_num):
        ret += " "

    return ret


def variable_to_code(var):
    """
    Get readable codes of fluid variable.

    Args:
        var: A fluid operator.

    Returns:
        string: The formatted string.
    """

    var_str = "{name} : fluid.{type}.shape{shape}.astype({dtype})".\
        format(i="{", e="}", name=var.name, type=var.type, shape=var.shape, dtype=var.dtype)

    if type(var) == paddle.fluid.framework.Parameter:
        if var.trainable:
            var_str = "trainable parameter " + var_str
        else:
            var_str = "parameter " + var_str
    else:
        var_str = "var " + var_str

    if var.persistable:
        var_str = "persist " + var_str

    return var_str


def op_to_code(op):
    """
    Get readable codes of fluid operator.

    Args:
        op: A fluid operator.

    Returns:
        string: The foramtted string.
    """

    outputs_str = "{"
    for i in range(0, len(op.output_names)):
        outputs_str += "{name}=".format(name=op.output_names[i])
        o = op.output(op.output_names[i])
        outputs_str += "{value}".format(value=o)
        if i != len(op.output_names) - 1:
            outputs_str += ", "
    outputs_str += "}"

    inputs_str = "{"
    for i in range(0, len(op.input_names)):
        inputs_str += "{name}=".format(name=op.input_names[i])
        o = op.input(op.input_names[i])
        inputs_str += "{value}".format(value=o)

        if i != len(op.input_names) - 1:
            inputs_str += ", "
    inputs_str += "}"

    attrs_str = ""
    for i in range(0, len(op.attr_names)):
        name = op.attr_names[i]

        attr_type = op.desc.attr_type(name)
        if attr_type == core.AttrType.BLOCK:
            a = "{name} = block[{value}]".format(
                name=name, type=attr_type, value=op.block_attr_id(name))
            attrs_str += a
            continue

        if attr_type == core.AttrType.BLOCKS:
            a = "{name} = blocks{value}".format(
                name=name, type=attr_type, value=op.blocks_attr_ids(name))
            attrs_str += a
            continue

        a = "{name} = {value}".format(
            name=name, type=attr_type, value=op.desc.attr(name))
        attrs_str += a
        if i != len(op.attr_names) - 1:
            attrs_str += ", "

    if outputs_str != "{}":
        op_str = "{outputs} = {op_type}(inputs={inputs}, {attrs})".\
            format(outputs = outputs_str, op_type=op.type, inputs=inputs_str, attrs=attrs_str)
    else:
        op_str = "{op_type}(inputs={inputs}, {attrs})".\
            format(op_type=op.type, inputs=inputs_str, attrs=attrs_str)
    return op_str


def program_to_code(prog):
    """
    Print readable codes of fluid program.

    Args:
        prog : A fluid program.

    An example result like bellow:
    https://github.com/PaddlePaddle/Paddle/pull/12673
    """
    indent = 0
    block_idx = 0
    for block in prog.blocks:
        print("{0}{1} // block {2}".format(
            get_indent_space(indent), '{', block_idx))
        indent += 1
        # sort all vars
        all_vars = sorted(block.vars.iteritems(), key=lambda x: x[0])
        for var in all_vars:
            print("{}{}".format(
                get_indent_space(indent), variable_to_code(var[1])))

        if len(all_vars) > 0:
            print("")

        for op in block.ops:
            print("{}{}".format(get_indent_space(indent), op_to_code(op)))
        indent -= 1
        print("{0}{1}".format(get_indent_space(indent), '}'))
        block_idx += 1
