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

# TODO: define normalization api  
import paddle
import paddle.fluid as fluid
from ...fluid.data_feeder import check_variable_and_dtype, check_type
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, core
from ...framework import create_parameter
from ...fluid.layers import l2_normalize  #DEFINE_ALIAS
from ...fluid.layers import lrn  #DEFINE_ALIAS
from ...fluid.initializer import Constant
from ...fluid.param_attr import ParamAttr
from ...fluid import core, dygraph_utils

__all__ = [
    'batch_norm',
    #       'data_norm',
    'instance_norm',
    'l2_normalize',
    'layer_norm',
    'lrn',
    'normalize',
    #       'spectral_norm'
]


def normalize(x, p=2, axis=1, epsilon=1e-12, name=None):
    """
    This op normalizes ``x`` along dimension ``axis`` using :math:`L_p` norm. This layer computes

    .. math::

        y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }
    
    .. math::
        \lvert \lvert x \rvert \rvert_p = \left(\sum_i {\lvert x_i\rvert^p}  \right)^{1/p}

    where, :math:`\sum_i{\lvert x_i\rvert^p}` is calculated along the ``axis`` dimension.


    Args:
        x (Tensor): The input tensor could be N-D tensor, and the input data type could be float32 or float64.
        p (float|int, optional): The exponent value in the norm formulation. Default: 2
        axis (int, optional): The axis on which to apply normalization. If ``x`` is 1-D tensor, ``axis`` is fixed to 0. If `axis < 0`, \
            the dimension to normalization is `x.ndim + axis`. -1 is the last dimension.
        epsilon (float, optional): Small float added to denominator to avoid dividing by zero. Default is 1e-12.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the output has the same shape and data type with ``x``.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn.functional as F

            paddle.disable_static()
            x = np.arange(6, dtype=np.float32).reshape(2,3)
            x = paddle.to_variable(x)
            y = F.normalize(x)
            print(y.numpy())
            # [[0.         0.4472136  0.8944272 ]
            # [0.42426404 0.5656854  0.7071067 ]]

            y = F.normalize(x, p=1.5)
            print(y.numpy())
            # [[0.         0.40862012 0.81724024]
            # [0.35684016 0.4757869  0.5947336 ]]

            y = F.normalize(x, axis=0)
            print(y.numpy())
            # [[0.         0.24253564 0.37139067]
            # [1.         0.97014254 0.9284767 ]]
    """
    if len(x.shape) == 1:
        axis = 0
    if in_dygraph_mode():
        eps = fluid.dygraph.base.to_variable([epsilon], dtype=x.dtype)
        out = core.ops.p_norm(x, 'axis', axis, 'porder',
                              float(p), 'keepdim', True, 'epsilon', epsilon)
        return x / core.ops.elementwise_max(out, eps)

    check_type(p, 'p', (float, int), 'normalize')
    check_type(axis, 'axis', (int), 'normalize')
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'normalize')

    attrs = {
        'axis': axis,
        'porder': float(p),
        'keepdim': True,
        'epsilon': epsilon,
    }
    helper = LayerHelper('p_norm', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='p_norm', inputs={'X': x}, outputs={'Out': out}, attrs=attrs)
    eps = out.block.create_var(dtype=out.dtype)
    paddle.fill_constant([1], out.dtype, epsilon, out=eps)
    return paddle.elementwise_div(x, paddle.maximum(out, eps), name=name)


def batch_norm(x,
               running_mean,
               running_var,
               weight,
               bias,
               training=False,
               momentum=0.9,
               epsilon=1e-05,
               data_format="NCHW",
               name=None):
    """
    Applies Batch Normalization as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    nn.functional.batch_norm is uesd for nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d. Please use above API for BatchNorm.
    
    Parameters:
        x(Tesnor): input value. It's data type should be float32, float64.
        running_mean(Tensor): running mean.
        running_var(Tensor): running variance.
        weight(Tensor): The weight tensor of batch_norm, can not be None.
        bias(Tensor): The bias tensor of batch_norm can not be None. 
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        training(bool, optional): True means train mode which compute by batch data and track global mean and var during train period. False means inference mode which compute by global mean and var which calculated by train period. Defalut False.
        data_format(str, optional): Specify the input data format. Defalut "NCHW".
        name(str, optional): Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          x = np.random.seed(123)
          x = np.random.random(size=(2, 1, 2, 3)).astype('float32')
          running_mean = np.random.random(size=1).astype('float32')
          running_variance = np.random.random(size=1).astype('float32')
          weight_data = np.random.random(size=1).astype('float32')
          bias_data = np.random.random(size=1).astype('float32')
          x = paddle.to_tensor(x)
          rm = paddle.to_tensor(running_mean)
          rv = paddle.to_tensor(running_variance)
          w = paddle.to_tensor(weight_data)
          b = paddle.to_tensor(bias_data)
          batch_norm_out = paddle.nn.functional.batch_norm(x, rm, rv, w, b)
          print batch_norm_out
    """

    assert len(x.shape) >= 2, "input dim must be larger than 1"

    # we use not training means use_global_status, more details see nn._BatchNormBase
    use_global_stats = not training
    # input ad out must share the memory
    mean_out = running_mean
    variance_out = running_var

    if in_dygraph_mode():
        # for dygraph need tuple
        attrs = ("momentum", momentum, "epsilon", epsilon, "data_layout",
                 data_format, "use_mkldnn", False, "fuse_with_relu", False,
                 "use_global_stats", use_global_stats)
        batch_norm_out, _, _, _, _, _ = core.ops.batch_norm(
            x, weight, bias, running_mean, running_var, mean_out, variance_out,
            *attrs)

        return dygraph_utils._append_activation_in_dygraph(
            batch_norm_out, act=None)

    check_variable_and_dtype(x, 'input', ['float16', 'float32', 'float64'],
                             'BatchNorm')

    # for static need dict
    attrs = {
        "momentum": momentum,
        "epsilon": epsilon,
        "data_layout": data_format,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats,
    }

    inputs = {
        "X": [x],
        "Scale": [weight],
        "Bias": [bias],
        "Mean": [running_mean],
        "Variance": [running_var]
    }

    helper = LayerHelper('batch_norm', **locals())

    dtype = x.dtype if x.dtype is not 'float16' else 'float32'
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    batch_norm_out = helper.create_variable_for_type_inference(dtype)

    outputs = {
        "Y": [batch_norm_out],
        "MeanOut": [running_mean],
        "VarianceOut": [running_var],
        "SavedMean": [saved_mean],
        "SavedVariance": [saved_variance]
    }

    helper.append_op(
        type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

    return helper.append_activation(batch_norm_out)


def layer_norm(x,
               normalized_shape,
               weight=None,
               bias=None,
               epsilon=1e-05,
               name=None):
    """
    see more detail in paddle.nn.LayerNorm
    
    Parameters:
        x(Tensor): Input Tensor. It's data type should be float32, float64.
        normalized_shape(int|list|tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight(Tensor, optional): The weight tensor of batch_norm. Default: None.
        bias(Tensor, optional): The bias tensor of batch_norm. Default: None.
        name(str, optional): Default None.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          layer_norm = paddle.nn.functional.layer_norm(x, x.shape[1:])
          layer_norm_out = layer_norm(x)

          print(layer_norm_out.numpy)
    """
    input_shape = list(x.shape)
    input_ndim = len(input_shape)
    normalized_ndim = len(normalized_shape)
    begin_norm_axis = input_ndim - normalized_ndim
    if input_ndim < normalized_ndim or input_shape[
            begin_norm_axis:] != normalized_shape:
        str_normalized_shape = str(normalized_shape)
        raise ValueError('Given normalized_shape is ' + str_normalized_shape +
                         ', expected input with shape [*, ' +
                         str_normalized_shape[
                             1:] + ', but got input shape ' + str(input_shape))

    if in_dygraph_mode():
        pre_act, _, _ = core.ops.layer_norm(x, weight, bias, 'epsilon', epsilon,
                                            'begin_norm_axis', begin_norm_axis)
        return dygraph_utils._append_activation_in_dygraph(pre_act, act=None)

    check_variable_and_dtype(x, 'input', ['float32', 'float64'], 'LayerNorm')

    inputs = dict()
    inputs['X'] = [x]
    if weight:
        inputs['Scale'] = [weight]
    if bias:
        inputs['Bias'] = [bias]
    attrs = {"epsilon": epsilon, "begin_norm_axis": begin_norm_axis}

    # create output
    helper = LayerHelper('layer_norm', **locals())
    mean_out = helper.create_variable_for_type_inference(
        dtype=x.type, stop_gradient=True)
    variance_out = helper.create_variable_for_type_inference(
        dtype=x.type, stop_gradient=True)
    layer_norm_out = helper.create_variable_for_type_inference(x.type)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})

    return helper.append_activation(layer_norm_out)


def instance_norm(x,
                  running_mean=None,
                  running_var=None,
                  weight=None,
                  bias=None,
                  use_input_stats=True,
                  momentum=0.9,
                  eps=1e-05,
                  data_format="NCHW",
                  name=None):
    """
    See more detail in nn.layer.InstanceNorm2d.

    Parameters:
        x(Tensor): Input Tensor. It's data type should be float32, float64.
        running_mean(Tensor): running mean. Default None.
        running_var(Tensor): running variance. Default None.
        weight(Tensor, optional): The weight tensor of instance_norm. Default: None.
        bias(Tensor, optional): The bias tensor of instance_norm. Default: None.
        eps(float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        use_input_stats(bool): Default True.
        data_format(str, optional): Specify the input data format. Default: NCHW.
        name(str, optional): Default None.

    Returns:
        None.

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          paddle.disable_static()
          np.random.seed(123)
          x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
          x = paddle.to_tensor(x_data) 
          instance_norm_out = paddle.nn.functional.instancenorm(x)

          print(instance_norm_out.numpy)

    """

    if in_dygraph_mode():
        out, _, _ = core.ops.instance_norm(x, weight, bias, "epsilon", eps,
                                           "momentum", momentum, "data_format",
                                           data_format)
        return out

    check_variable_and_dtype(x, 'input', ['float32', 'float64'], "InstanceNorm")

    attrs = {"epsilon": eps, "momentum": momentum, "data_format": data_format}

    if weight and bias:
        inputs = {"X": [x], "Scale": [weight], "Bias": [bias]}
    else:
        inputs = {"X": [x]}

    helper = LayerHelper('instance_norm', **locals())
    saved_mean = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True)
    instance_norm_out = helper.create_variable_for_type_inference(x.dtype)

    outputs = {
        "Y": [instance_norm_out],
        "SavedMean": [saved_mean],
        "SavedVariance": [saved_variance]
    }

    helper.append_op(
        type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs)
    return instance_norm_out
