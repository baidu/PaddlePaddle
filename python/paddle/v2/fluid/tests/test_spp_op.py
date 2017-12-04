import unittest
import numpy as np
from op_test import OpTest
from test_pool2d_op import max_pool2D_forward_naive


class TestSppOp(OpTest):
    def setUp(self):
        self.op_type = "spp"
        self.init_test_case()
        input = np.random.random(self.shape).astype("float32")
        nsize, csize, hsize, wsize = input.shape
        out_level_flatten = []
        for i in xrange(self.pyramid_height):
            bins = np.power(2, i)
            kernel_size = [0, 0]
            padding = [0, 0]
            kernel_size[0] = np.ceil(hsize /
                                     bins.astype("double")).astype("int32")
            padding[0] = (
                (kernel_size[0] * bins - hsize + 1) / 2).astype("int32")

            kernel_size[1] = np.ceil(wsize /
                                     bins.astype("double")).astype("int32")
            padding[1] = (
                (kernel_size[1] * bins - wsize + 1) / 2).astype("int32")
            out_level = max_pool2D_forward_naive(input, kernel_size,
                                                 kernel_size, padding)
            out_level_flatten.append(
                out_level.reshape(nsize, bins * bins * csize))
            if i == 0:
                output = out_level_flatten[i]
            else:
                output = np.concatenate((output, out_level_flatten[i]), 1)
        # output = np.concatenate(out_level_flatten.tolist(), 0);
        self.inputs = {'X': input.astype('float32'), }
        self.attrs = {'pyramid_height': self.pyramid_height}

        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.05)

    def init_test_case(self):
        self.shape = [3, 2, 4, 4]
        self.pyramid_height = 3


if __name__ == '__main__':
    unittest.main()
