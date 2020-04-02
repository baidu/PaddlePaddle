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
from six.moves import reduce
from ..fluid.layer_helper import LayerHelper
from ..fluid.param_attr import ParamAttr
from ..fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator
from ..fluid.framework import Variable, device_guard
from ..fluid.initializer import Constant
from ..fluid.layers.layer_function_generator import templatedoc
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from ..fluid.layers import utils
import numpy
import warnings
from ..fluid.layers.tensor import fill_constant

# TODO: define random functions  
__all__ = [
    'randint'
    #            'gaussin', 
    #            'uniform', 
    #            'shuffle',
    #            'randn',
    #            'randperm',
    #            'rand',
]


def randint(low,
            high=None,
            shape=None,
            out=None,
            dtype=None,
            device=None,
            stop_gradient=False,
            name=None):
    """
    This function returns a Tensor filled with random integers from the "discrete uniform" distribution of the
    specified dtype in the interval [low, high). If high is None (the default), then results are from [0, low).

    Args:
        low (int): The lower bound on the range of random values to generate, the low is included in the range.
            (unless high=None, in which case this parameter is one above the highest such integer).
        high (int, optional): The upper bound on the range of random values to generate, the high is excluded 
            in the range. Default None(see above for behavior if high=None).
        shape (list|tuple|Variable, optional): The shape of the output Tensor,  if the shape is a list or tuple, 
                                     its elements can be an integer
                                     or a Tensor with the shape [1], and the type of the Tensor must be int32 or int64. 
                                     If the shape is a Variable, it is a 1-D Tensor, and the type of the Tensor must be 
                                     int32 or int64. Default is None, in which case the shape is [1].
        out(Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            if out is None, a new Varibale will be create to store the result.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output Tensor
            which can be float32, float64, int32, int64, if dytpe is `None`, the data
            type of created Tensor is `int32`
        device(str, optional): This parameter specifies that the Tensor is created 
            on the GPU or CPU.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) Variable,
            default value is False.
        name(str, optional): The default value is None.  Normally there is no need for user to set this
            property.  For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python
            import paddle
            import paddle.tensor as tensor

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.randint(low=-5, high=5, shape=[3, 4], dtype="int64")

            # example 2:
            # attr shape is a list which contains tensor Variable.
            dim_1 = fluid.layers.fill_constant([1],"int64",3)
            dim_2 = fluid.layers.fill_constant([1],"int32",5)
            result_2 = paddle.randint(low=-5, high=5, shape=[dim_1, dim_2], dtype="int32")

            # example 3:
            # attr shape is a Variable, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = padddle.randint(low=-5, high=5, shape=var_shape, dtype="float32")
            var_shape_int32 = fluid.data(name='var_shape_int32', shape=[2], dtype="int32")
            result_4 = paddle.randint(low=-5, high=5, shape=var_shape_int32, dtype="float64")

            # example 4:
            # Input only one parameter
            # low=0, high=10, shape=[1], dtype='int32'
            result_4 = paddle.randint(10)
     """

    def get_new_shape_tensor(list_shape):
        new_shape_tensor = []
        for dim in list_shape:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_shape_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int64')
                fill_constant([1], 'int64', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                assert dim_size > 0, (
                    "Each dimension size given in shape must not be negative "
                    "except one unknown dimension.")
        return attrs_shape

    if dtype is None:
        dtype = 'int32'
    check_dtype(dtype, 'dtype', ['int32', 'int64', 'float32', 'float64'],
                'randint')

    inputs = dict()
    attrs = dict()

    if shape is None:
        shape = [1]
        assert len(shape) > 0, ("The size of argument(shape) can't be zero.")

    helper = LayerHelper("randint", **locals())

    if in_dygraph_mode():
        attrs['shape'] = shape
    else:
        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs["ShapeTensor"] = shape
        elif isinstance(shape, (list, tuple)):
            assert len(shape) > 0, (
                "The size of argument(shape) can't be zero.")
            if utils._contain_var(shape):
                inputs['ShapeTensorList'] = get_new_shape_tensor(shape)
            else:
                attrs["shape"] = get_attr_shape(shape)
    check_type(shape, 'shape', (list, tuple, Variable), 'randint')

    if high is None:
        high = low
        low = 0
    attrs['low'] = low
    attrs['high'] = high

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=dtype, persistable=False)
    else:
        check_dtype(dtype, 'dtype',
                    convert_dtype(out.dtype), 'randint',
                    "(The dtype in randint must be the same with out's dtype.)")
    attrs['dtype'] = out.dtype
    out.stop_gradient = stop_gradient

    if device is None:
        helper.append_op(
            type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    else:
        with device_guard(device):
            helper.append_op(
                type='randint',
                inputs=inputs,
                outputs={'Out': out},
                attrs=attrs)
    return out
