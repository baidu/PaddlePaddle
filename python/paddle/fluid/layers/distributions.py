# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import control_flow
from . import tensor
from . import ops
from . import nn
import math
import numpy as np
import warnings

__all__ = ['Uniform', 'Normal', 'Categorical', 'MultivariateNormalDiag']


class Distribution(object):
    """
    Distribution is the abstract base class for probability distributions.
    """

    def sample(self):
        """Sampling from the distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError

    def kl_divergence(self, other):
        """The KL-divergence between self distributions and other."""
        raise NotImplementedError

    def log_prob(self, value):
        """Log probability density/mass function."""
        raise NotImplementedError

    def _validate_args(self, *args):
        """
        Argument validation for distribution args
        Args:
            value (float, list, numpy.ndarray, Variable)
        Raises
            ValueError: if one argument is Variable, all arguments should be Variable
        """
        is_variable = False
        is_number = False
        for arg in args:
            if isinstance(arg, tensor.Variable):
                is_variable = True
            else:
                is_number = True

        if is_variable and is_number:
            raise ValueError(
                'if one argument is Variable, all arguments should be Variable')

        return is_variable

    def _to_variable(self, *args):
        """
        Argument convert args to Variable

        Args:
            value (float, list, numpy.ndarray, Variable)
        Returns:
            Variable of args.
        """
        numpy_args = []
        variable_args = []
        tmp = 0.

        for arg in args:
            valid_arg = False
            for cls in [float, list, np.ndarray, tensor.Variable]:
                if isinstance(arg, cls):
                    valid_arg = True
                    break
            assert valid_arg, "type of input args must be float, list, numpy.ndarray or Variable."
            if isinstance(arg, float):
                arg = np.zeros(1) + arg
            arg_np = np.array(arg)
            arg_dtype = arg_np.dtype
            if str(arg_dtype) not in ['float32']:
                warnings.warn(
                    "data type of argument only support float32, your argument will be convert to float32."
                )
                arg_np = arg_np.astype('float32')
            tmp = tmp + arg_np
            numpy_args.append(arg_np)

        dtype = tmp.dtype
        for arg in numpy_args:
            arg_broadcasted, _ = np.broadcast_arrays(arg, tmp)
            arg_variable = tensor.create_tensor(dtype=dtype)
            tensor.assign(arg_broadcasted, arg_variable)
            variable_args.append(arg_variable)

        return tuple(variable_args)


class Uniform(Distribution):
    """Uniform distribution with `low` and `high` parameters.

    Mathematical Details

    The probability density function (pdf) is,

    .. math::

        pdf(x; a, b) = \\frac{1}{Z}, \ a <=x <b

    .. math::

        Z = b - a

    In the above equation:

    * :math:`low = a`,
    * :math:`high = b`,
    * :math:`Z`: is the normalizing constant.

    The parameters `low` and `high` must be shaped in a way that supports
    broadcasting (e.g., `high - low` is a valid operation).

    Args:
        low(float|list|numpy.ndarray|Variable): The lower boundary of uniform distribution.
        high(float|list|numpy.ndarray|Variable): The higher boundary of uniform distribution.

    Examples:
        .. code-block:: python

          from paddle.fluid import layers
          from paddle.fluid.layers import Uniform

          # Without broadcasting, a single uniform distribution [3, 4]:
          u1 = Uniform(low=3.0, high=4.0)
          # 2 distributions [1, 3], [2, 4]
          u2 = Uniform(low=[1.0, 2.0],
                        high=[3.0, 4.0])
          # 4 distributions
          u3 = Uniform(low=[[1.0, 2.0],
                    [3.0, 4.0]],
               high=[[1.5, 2.5],
                     [3.5, 4.5]])

          # With broadcasting:
          u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

          # Variable as input
          dims = 3

          low = layers.data(name='low', shape=[dims], dtype='float32')
          high = layers.data(name='high', shape=[dims], dtype='float32')
          values = layers.data(name='values', shape=[dims], dtype='float32')

          uniform = Uniform(low, high)

          sample = uniform.sample([2, 3])
          entropy = uniform.entropy()
          lp = uniform.log_prob(values)

    """

    def __init__(self, low, high):
        self.all_arg_is_float = False
        self.batch_size_unknown = False
        if self._validate_args(low, high):
            self.batch_size_unknown = True
            self.low = low
            self.high = high
        else:
            if isinstance(low, float) and isinstance(high, float):
                self.all_arg_is_float = True
            self.low, self.high = self._to_variable(low, high)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Variable: A tensor with prepended dimensions shape.

        """
        batch_shape = list((self.low + self.high).shape)
        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.low + self.high, batch_shape + shape, self.low.dtype, 0.)
            uniform_random_tmp = nn.uniform_random_batch_size_like(
                zero_tmp, zero_tmp.shape, min=0., max=1., seed=seed)
            output = uniform_random_tmp * (zero_tmp + self.high - self.low
                                           ) + self.low
            return nn.reshape(output, output_shape)
        else:
            output_shape = shape + batch_shape
            output = ops.uniform_random(
                output_shape, seed=seed) * (tensor.zeros(
                    output_shape, dtype=self.low.dtype) +
                                            (self.high - self.low)) + self.low
            if self.all_arg_is_float:
                return nn.reshape(output, shape)
            else:
                return output

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Variable): The input tensor.

        Returns:
          Variable: log probability.

        """
        lb_bool = control_flow.less_than(self.low, value)
        ub_bool = control_flow.less_than(value, self.high)
        lb = tensor.cast(lb_bool, dtype=value.dtype)
        ub = tensor.cast(ub_bool, dtype=value.dtype)
        return nn.log(lb * ub) - nn.log(self.high - self.low)

    def entropy(self):
        """Shannon entropy in nats.

        Returns:
          Variable: Shannon entropy of uniform distribution.

        """
        return nn.log(self.high - self.low)


class Normal(Distribution):
    """The Normal distribution with location `loc` and `scale` parameters.

    Mathematical details

    The probability density function (pdf) is,

    .. math::

        pdf(x; \mu, \sigma) = \\frac{1}{Z}e^{\\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    .. math::

        Z = (2 \pi \sigma^2)^{0.5}

    In the above equation:

    * :math:`loc = \mu`: is the mean.
    * :math:`scale = \sigma`: is the std.
    * :math:`Z`: is the normalization constant.

    Args:
        loc(float|list|numpy.ndarray|Variable): The mean of normal distribution.
        scale(float|list|numpy.ndarray|Variable): The std of normal distribution.

    Examples:
        .. code-block:: python

          from paddle.fluid import layers
          from paddle.fluid.layers import Normal

          # Define a single scalar Normal distribution.
          dist = Normal(loc=0., scale=3.)
          # Define a batch of two scalar valued Normals.
          # The first has mean 1 and standard deviation 11, the second 2 and 22.
          dist = Normal(loc=[1, 2.], scale=[11, 22.])
          # Get 3 samples, returning a 3 x 2 tensor.
          dist.sample([3])

          # Define a batch of two scalar valued Normals.
          # Both have mean 1, but different standard deviations.
          dist = Normal(loc=1., scale=[11, 22.])

          # Define a batch of two scalar valued Normals.
          # Both have mean 1, but different standard deviations.
          dist = Normal(loc=1., scale=[11, 22.])

          # Variable as input
          dims = 3

          loc = layers.data(name='loc', shape=[dims], dtype='float32')
          scale = layers.data(name='scale', shape=[dims], dtype='float32')
          other_loc = layers.data(
              name='other_loc', shape=[dims], dtype='float32')
          other_scale = layers.data(
              name='other_scale', shape=[dims], dtype='float32')
          values = layers.data(name='values', shape=[dims], dtype='float32')

          normal = Normal(loc, scale)
          other_normal = Normal(other_loc, other_scale)

          sample = normal.sample([2, 3])
          entropy = normal.entropy()
          lp = normal.log_prob(values)
          kl = normal.kl_divergence(other_normal)
    """

    def __init__(self, loc, scale):
        self.batch_size_unknown = False
        self.all_arg_is_float = False
        if self._validate_args(loc, scale):
            self.batch_size_unknown = True
            self.loc = loc
            self.scale = scale
        else:
            if isinstance(loc, float) and isinstance(scale, float):
                self.all_arg_is_float = True
            self.loc, self.scale = self._to_variable(loc, scale)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Variable: A tensor with prepended dimensions shape.

        """
        batch_shape = list((self.loc + self.scale).shape)

        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.loc + self.scale, batch_shape + shape, self.loc.dtype, 0.)
            normal_random_tmp = nn.gaussian_random_batch_size_like(
                zero_tmp, zero_tmp.shape, mean=0., std=1., seed=seed)
            output = normal_random_tmp * (zero_tmp + self.scale) + self.loc
            return nn.reshape(output, output_shape)
        else:
            output_shape = shape + batch_shape
            output = nn.gaussian_random(output_shape, mean=0., std=1., seed=seed) * \
                     (tensor.zeros(output_shape, dtype=self.loc.dtype) + self.scale) + self.loc
            if self.all_arg_is_float:
                return nn.reshape(output, shape)
            else:
                return output

    def entropy(self):
        """Shannon entropy in nats.

        Returns:
          Variable: Shannon entropy of normal distribution.

        """
        batch_shape = list((self.loc + self.scale).shape)
        zero_tmp = tensor.fill_constant_batch_size_like(
            self.loc + self.scale, batch_shape, self.loc.dtype, 0.)
        return 0.5 + 0.5 * math.log(2 * math.pi) + nn.log(
            (self.scale + zero_tmp))

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Variable): The input tensor.

        Returns:
          Variable: log probability.

        """
        var = self.scale * self.scale
        log_scale = nn.log(self.scale)
        return -1. * ((value - self.loc) * (value - self.loc)) / (
            2. * var) - log_scale - math.log(math.sqrt(2. * math.pi))

    def kl_divergence(self, other):
        """The KL-divergence between two normal distributions.

        Args:
            other (Normal): instance of Normal.

        Returns:
            Variable: kl-divergence between two normal distributions.

        """
        assert isinstance(other, Normal), "another distribution must be Normal"
        var_ratio = self.scale / other.scale
        var_ratio = (var_ratio * var_ratio)
        t1 = (self.loc - other.loc) / other.scale
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1. - nn.log(var_ratio))


class Categorical(Distribution):
    """
    Categorical distribution is a discrete probability distribution that 
    describes the possible results of a random variable that can take on 
    one of K possible categories, with the probability of each category 
    separately specified.

    Args:
        logits(list|numpy.ndarray|Variable): The logits input of categorical distribution.

    Examples:
        .. code-block:: python

          import numpy as np
          from paddle.fluid import layers
          from paddle.fluid.layers import Categorical

          a_logits_npdata = np.array([-0.602,-0.602], dtype="float32")
          a_logits_tensor = layers.create_tensor(dtype="float32")
          layers.assign(a_logits_npdata, a_logits_tensor)

          b_logits_npdata = np.array([-0.102,-0.112], dtype="float32")
          b_logits_tensor = layers.create_tensor(dtype="float32")
          layers.assign(b_logits_npdata, b_logits_tensor)
          
          a = Categorical(a_logits_tensor)
          b = Categorical(b_logits_tensor)

          a.entropy()
          b.entropy()
          
          a.kl_divergence(b)

    """

    def __init__(self, logits):
        """
        Args:
            logits: A float32 tensor
        """
        if self._validate_args(logits):
            self.logits = logits
        else:
            self.logits = self._to_variable(logits)[0]

    def kl_divergence(self, other):
        """The KL-divergence between two Categorical distributions.

        Args:
            other (Categorical): instance of Categorical.

        Returns:
            Variable: kl-divergence between two Categorical distributions.

        """
        assert isinstance(other, Categorical)

        logits = self.logits - nn.reduce_max(self.logits, dim=-1, keep_dim=True)
        other_logits = other.logits - nn.reduce_max(
            other.logits, dim=-1, keep_dim=True)
        e_logits = ops.exp(logits)
        other_e_logits = ops.exp(other_logits)
        z = nn.reduce_sum(e_logits, dim=-1, keep_dim=True)
        other_z = nn.reduce_sum(other_e_logits, dim=-1, keep_dim=True)
        prob = e_logits / z
        kl = nn.reduce_sum(
            prob * (logits - nn.log(z) - other_logits + nn.log(other_z)),
            dim=-1,
            keep_dim=True)

        return kl

    def entropy(self):
        """Shannon entropy in nats.

        Returns:
          Variable: Shannon entropy of Categorical distribution.

        """
        logits = self.logits - nn.reduce_max(self.logits, dim=-1, keep_dim=True)
        e_logits = ops.exp(logits)
        z = nn.reduce_sum(e_logits, dim=-1, keep_dim=True)
        prob = e_logits / z
        entropy = -1.0 * nn.reduce_sum(
            prob * (logits - nn.log(z)), dim=-1, keep_dim=True)

        return entropy


class MultivariateNormalDiag(Distribution):
    """
    A multivariate normal (also called Gaussian) distribution parameterized by a mean vector
    and a covariance matrix.

    Args:
        loc(list|numpy.ndarray|Variable): The mean of multivariateNormal distribution.
        scale(list|numpy.ndarray|Variable): The positive definite diagonal covariance matrix of
        multivariateNormal distribution.

    Examples:
        .. code-block:: python
    
            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid.layers import MultivariateNormalDiag

            a_loc_npdata = np.array([0.3,0.5],dtype="float32")
            a_loc_tensor = layers.create_tensor(dtype="float32")
            layers.assign(a_loc_npdata, a_loc_tensor)


            a_scale_npdata = np.array([[0.4,0],[0,0.5]],dtype="float32")
            a_scale_tensor = layers.create_tensor(dtype="float32")
            layers.assign(a_scale_npdata, a_scale_tensor)

            b_loc_npdata = np.array([0.2,0.4],dtype="float32")
            b_loc_tensor = layers.create_tensor(dtype="float32")
            layers.assign(b_loc_npdata, b_loc_tensor)

            b_scale_npdata = np.array([[0.3,0],[0,0.4]],dtype="float32")
            b_scale_tensor = layers.create_tensor(dtype="float32")
            layers.assign(b_scale_npdata, b_scale_tensor)

            a = MultivariateNormalDiag(a_loc_tensor, a_scale_tensor)
            b = MultivariateNormalDiag(b_loc_tensor, b_scale_tensor)
            
            a.entropy()
            b.entropy()

            a.kl_divergence(b)
       
    """

    def __init__(self, loc, scale):
        if self._validate_args(loc, scale):
            self.loc = loc
            self.scale = scale
        else:
            self.loc, self.scale = self._to_variable(loc, scale)

    def _det(self, value):

        batch_shape = list(value.shape)
        one_all = tensor.ones(shape=batch_shape, dtype=self.loc.dtype)
        one_diag = tensor.diag(
            tensor.ones(
                shape=[batch_shape[0]], dtype=self.loc.dtype))
        det_diag = nn.reduce_prod(value + one_all - one_diag)

        return det_diag

    def _inv(self, value):

        batch_shape = list(value.shape)
        one_all = tensor.ones(shape=batch_shape, dtype=self.loc.dtype)
        one_diag = tensor.diag(
            tensor.ones(
                shape=[batch_shape[0]], dtype=self.loc.dtype))
        inv_diag = nn.elementwise_pow(value, (one_all - 2 * one_diag))

        return inv_diag

    def entropy(self):
        """Shannon entropy in nats.

        Returns:
          Variable: Shannon entropy of Multivariate Normal distribution.

        """
        entropy = 0.5 * (
            self.scale.shape[0] *
            (1.0 + math.log(2 * math.pi)) + nn.log(self._det(self.scale)))

        return entropy

    def kl_divergence(self, other):
        """The KL-divergence between two Multivariate Normal distributions.

        Args:
            other (MultivariateNormalDiag): instance of Multivariate Normal.

        Returns:
            Variable: kl-divergence between two Multivariate Normal distributions.

        """
        assert isinstance(other, MultivariateNormalDiag)

        tr_cov_matmul = nn.reduce_sum(self._inv(other.scale) * self.scale)
        loc_matmul_cov = nn.matmul((other.loc - self.loc),
                                   self._inv(other.scale))
        tri_matmul = nn.matmul(loc_matmul_cov, (other.loc - self.loc))
        k = list(self.scale.shape)[0]
        ln_cov = nn.log(self._det(other.scale)) - nn.log(self._det(self.scale))
        kl = 0.5 * (tr_cov_matmul + tri_matmul - k + ln_cov)

        return kl
