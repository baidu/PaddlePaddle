import unittest
import numpy as np
from op_test import OpTest


class TestAdamOp1(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOp2(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        attributes = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adam Operator with supplied attributes
        '''
        self.op_type = "adam"
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment1_out, \
                moment2_out = adam_step(self.inputs, self.attrs)

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment1'] = moment1_out
            self.inputs['Moment2'] = moment2_out

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] *= self.attrs['beta1']
            self.inputs['Beta2Pow'] *= self.attrs['beta1']

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(
                -1, 1, (102, 105)).astype("float32")


def adam_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


def adam_step_sparse(inputs, attributes, height, rows, row_numel, np_grad):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    # grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment1_out = np.array([height, row_numel])
    moment2_out = np.array([height, row_numel])
    param_out = np.array([height, row_numel])

    for idx, row_id in enumerate(rows):
        moment1_out[row_id] = beta1 * moment1[row_id] + (1 - beta1
                                                         ) * np_grad[idx]
        moment2_out[row_id] = beta2 * moment2[row_id] + (
            1 - beta2) * np.square(np_grad[idx])
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
        param_out[row_id] = param[row_id] - lr_t * (moment1_out / (
            np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


class TestSparseAdamOp(unittest.TestCase):
    def setup(self, scope, place):
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4

        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        self.dense_inputs = {
            "Param": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment1": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment2": np.full((height, row_numel), 5.0).astype("float32"),
            'Beta1Pow': np.array([0.9**10]).astype("float32"),
            'Beta2Pow': np.array([0.999**10]).astype("float32"),
            "LearningRate": np.full((1), 2.0).astype("float32")
        }
        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        self.sparse_inputs = ["Grad"]

        param_out, mom1, mom2 = adam_step_sparse(
            self.dense_inputs, self.attrs, height, rows, row_numel, np_array)
        self.outputs = {
            "Param": param_out,
            "Moment1Out": mom1,
            "Moment2Out": mom2
        }

    def check_with_place(self, place):
        scope = core.Scope()
        self.setup(scope, place)

        op_args = dict()
        for key, np_array in self.dense_inputs.iteritems():
            var = scope.var(key).get_tensor()
            var.set(np_array, place)
            op_args[key] = key
        for s in self.sparse_inputs:
            op_args[s] = s
        for k in self.attrs:
            op_args[k] = self.attrs[k]

        # create and run sgd operator
        sgd_op = Operator("adam", **op_args)
        sgd_op.run(scope, place)

        for key, np_array in self.outputs.iteritems():
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual.reshape([actual.size()])
            np_array.reshape([np_array.size()])
            i = 0
            while i < actual.size():
                self.assertAlmostEqual(actual[i], np_array[i])
                i += 1

        # # rows[0] = 0, 5.0 - 2.0 * 2.0
        # self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        # # rows[0] = 0, 5.0 - 2.0 * 1.0
        # self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        # # 5.0 - 2.0 * 0.0
        # self.assertAlmostEqual(5.0, result_array[1, 0])
        # # rows[1] = 4, 5.0 - 2.0 * 1.0
        # self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        # # 5.0 - 2.0 * 0.0
        # self.assertAlmostEqual(5.0, result_array[5, 8])
        # # rows[2] = 7, 5.0 - 2.0 * 1.0
        # self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        # # rows[2] = 7, 5.0 - 2.0 * 4.0
        # self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
