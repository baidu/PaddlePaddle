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

import copy
import six

from .framework import Parameter, dtype_is_floating, in_dygraph_mode, OpProtoHolder
from . import unique_name
from paddle.fluid.initializer import Constant, Xavier
from .param_attr import ParamAttr
from . import core
from six.moves import zip
from .layer_helper_base import LayerHelperBase


class LayerHelper(LayerHelperBase):
    def __init__(self, layer_type, **kwargs):
        self.kwargs = kwargs
        name = self.kwargs.get('name', None)
        # TODO(panyx0718, minqiyang): dygraph mode
        # can not use both `layer_type` and `name`. Deprecate LayerHelper
        # and write a Helper for dygraph mode.
        if name is None:
            self.kwargs['name'] = unique_name.generate(layer_type)

        super(LayerHelper, self).__init__(
            self.kwargs['name'], layer_type=layer_type)

    def append_op(self, *args, **kwargs):
        # Add log for api benchmark. may need a flag to control
        if os.environ.get('ENABLE_API_BENCH_LOG'):
            print('op_name:', kwargs['type'])
            if 'inputs' in kwargs:
                for key, values in kwargs['inputs'].items():
                    if key != "":
                        input_type = ""
                        data_type = ""
                        if isinstance(values, list):
                            values = values[0]
                        if hasattr(values, 'dtype'):
                            if values.dtype == core.VarDesc.VarType.FP32:
                                data_type = 'float32'
                            if values.dtype == core.VarDesc.VarType.FP64:
                                data_type = 'float64'
                            if values.dtype == core.VarDesc.VarType.INT32:
                                data_type = 'int32'
                            if values.dtype == core.VarDesc.VarType.INT64:
                                data_type = 'int64'
                            if values.dtype == core.VarDesc.VarType.INT8:
                                data_type = 'int8'
                            if type(values.type) is core.VarDesc.VarType:
                                input_type = "Variable"
                        else:
                            data_type = 'None'
                        print(key, ' ', input_type, "|", data_type, "|shape:",
                              list(values.shape))
            if 'attrs' in kwargs:
                for key, values in kwargs['attrs'].items():
                    vtype = ""
                    if isinstance(values, list):
                        values = values[0]
                    if type(values) is int:
                        vtype = "int"
                    if type(values) is list:
                        vtype = "list"
                    if type(values) is float:
                        vtype = "float"
                    if type(values) is bool:
                        vtype = "bool"
                    if type(values) is str:
                        vtype = "string"
                    if type(values) is core.VarDesc.VarType:
                        vtype = "string"
                        values = str(values).split('.')[-1]
                    print(key, ' ', vtype, '|', values)

        return self.main_program.current_block().append_op(*args, **kwargs)

    def multiple_input(self, input_param_name='input'):
        inputs = self.kwargs.get(input_param_name, [])
        ret = []
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            for inp in inputs:
                ret.append(self.to_variable(inp))
        else:
            ret.append(self.to_variable(inputs))
        return ret

    def input(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        if len(inputs) != 1:
            raise "{0} layer only takes one input".format(self.layer_type)
        return inputs[0]

    @property
    def param_attr(self):
        return ParamAttr._to_attr(self.kwargs.get('param_attr', None))

    @property
    def bias_attr(self):
        return ParamAttr._to_attr(self.kwargs.get('bias_attr', None))

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of param_attr
    def multiple_param_attr(self, length):
        param_attr = self.param_attr
        if isinstance(param_attr, ParamAttr):
            param_attr = [param_attr]

        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError("parameter number mismatch")
        elif len(param_attr) == 1 and length != 1:
            tmp = [None] * length
            for i in six.moves.range(length):
                tmp[i] = copy.deepcopy(param_attr[0])
            param_attr = tmp
        return param_attr

    def iter_inputs_and_params(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        param_attrs = self.multiple_param_attr(len(inputs))
        for ipt, param_attr in zip(inputs, param_attrs):
            yield ipt, param_attr

    def input_dtype(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.dtype
            elif dtype != each.dtype:
                raise ValueError("Data Type mismatch: %d to %d" %
                                 (dtype, each.dtype))
        return dtype

    def get_parameter(self, name):
        param = self.main_program.global_block().var(name)
        if not isinstance(param, Parameter):
            raise ValueError("no Parameter name %s found" % name)
        return param

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of bias_attr
    def append_bias_op(self, input_var, dim_start=1, dim_end=None):
        """
        Append bias operator and return its output. If the user does not set
        bias_attr, append_bias_op will return input_var

        :param input_var: the input variable. The len(input_var.shape) is
        larger or equal than 2.
        :bias_initializer: an instance of a subclass of Initializer used to
        initialize the bias
        :param dim_start:
        :param dim_end: the shape of the bias will be
        input_var.shape[dim_start:dim_end]. The bias is broadcasted to other
        dimensions and added to input_var to get the output
        """
        size = list(input_var.shape[dim_start:dim_end])
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var

        b = self.create_parameter(
            attr=bias_attr, shape=size, dtype=input_var.dtype, is_bias=True)
        tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
        self.append_op(
            type='elementwise_add',
            inputs={'X': [input_var],
                    'Y': [b]},
            outputs={'Out': [tmp]},
            attrs={'axis': dim_start})
        return tmp

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of act
    def append_activation(self, input_var):
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var
        if isinstance(act, six.string_types):
            act = {'type': act}
        else:
            raise TypeError(str(act) + " should be unicode or str")

        if 'use_cudnn' in self.kwargs and self.kwargs.get('use_cudnn'):
            act['use_cudnn'] = self.kwargs.get('use_cudnn')
        if 'use_mkldnn' in self.kwargs:
            act['use_mkldnn'] = self.kwargs.get('use_mkldnn')
        act_type = act.pop('type')

        tmp = self.create_variable_for_type_inference(dtype=input_var.dtype)
        self.append_op(
            type=act_type,
            inputs={"X": [input_var]},
            outputs={"Out": [tmp]},
            attrs=act)
        return tmp

    #TODO (jiabin): should we remove this since it has never be used
    def _get_default_initializer(self, dtype):
        if dtype is None or dtype_is_floating(dtype) is True:
            return Xavier()
        else:
            # For integer and boolean types, initialize with all zeros
            return Constant()

    #TODO (jiabin): reconstruct this in LayerObjHelper and avoid dependency of kwargs
    def is_instance(self, param_name, cls):
        param = self.kwargs.get(param_name, None)
        if not isinstance(param, cls):
            raise TypeError("The input {0} parameter of method {1} must be {2}",
                            param_name, self.layer_type, cls.__name__)
