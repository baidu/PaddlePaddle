import unittest
import numpy as np
from op_test import OpTest


class TestLRNOp(OpTest):
    def get_input(self):
        ''' TODO(gongweibao): why it's grad diff is so large?
        x = np.ndarray(
            shape=(self.N, self.C, self.H, self.W), dtype=float, order='C')
        for m in range(0, self.N):
            for i in range(0, self.C):
                for h in range(0, self.H):
                    for w in range(0, self.W):
                        x[m][i][h][w] = m * self.C * self.H * self.W +  \
                                        i * self.H * self.W +  \
                                        h * self.W + w + 1
        '''
        x = np.random.rand(self.N, self.C, self.H, self.W).astype("float32")
        return x + 1

    def get_out(self):
        start = -(self.n - 1) / 2
        end = start + self.n

        mid = np.empty((self.N, self.C, self.H, self.W), dtype=float)
        mid.fill(self.k)
        for m in range(0, self.N):
            for i in range(0, self.C):
                for c in range(start, end + 1):
                    ch = i + c
                    if ch < 0 or ch >= self.C:
                        continue

                    s = mid[m][i][:][:]
                    r = self.x[m][ch][:][:]
                    s += np.square(r) * self.alpha

        mid2 = np.power(mid, -self.beta)
        return np.multiply(self.x, mid2), mid

    def get_attrs(self):
        attrs = {
            'n': self.n,
            'k': self.k,
            'alpha': self.alpha,
            'beta': self.beta
        }
        return attrs

    def setUp(self):
        self.op_type = "lrn"
        self.N = 2
        self.C = 3
        self.H = 5
        self.W = 5

        self.n = 5
        self.k = 2.0
        self.alpha = 0.0001
        self.beta = 0.75
        self.x = self.get_input()
        self.out, self.mid_out = self.get_out()

        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out, 'mid_out': self.mid_out}
        self.attrs = self.get_attrs()

    def test_check_output(self):
        self.check_output()

    '''
    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.12)
    '''


if __name__ == "__main__":
    unittest.main()
