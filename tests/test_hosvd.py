import sys
import unittest

import numpy as np

sys.path.append('./')

from source.hosvd import hosvd, hosvd_dense, tucker2_dense
from source.tensor_operations import (
    gen_hilbert_tensor,
    sqrt_err_relative,
    construct_tensor_from_tucker,
    dense_to_sparse_tensor,
    sparse_to_dense_tensor,
)

class TestHOSVD(unittest.TestCase):
    def test_bad_rank(self):
        with self.assertRaises(RuntimeError):
            bad_rank = (5, 2, 2)

            shape = (20, 20, 20)
            inds, vals = gen_hilbert_tensor(shape)
            hosvd(
                inds, vals, shape=shape,
                rank=bad_rank, seed=0, verbose=False,
            )

    def test_hilbert_tensor(self):
        # prepare the data:
        shape = (20, 20, 20)
        inds, vals = gen_hilbert_tensor(shape)
        r = 9
        rank = (r, r, r)
        u0, u1, u2, core = hosvd(
            inds, vals, shape=shape,
            rank=rank, seed=0, verbose=False,
        )

        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0", places=4)

    def test_hilbert_tensor_dense(self):
        # prepare the data:
        shape = (20, 20, 20)
        inds, vals = gen_hilbert_tensor(shape)
        x = sparse_to_dense_tensor(inds, vals, shape)
        r = 9
        rank = (r, r, r)
        u0, u1, u2, core = hosvd_dense(
            x, rank=rank, seed=0, verbose=False,
        )

        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0", places=4)    

    def test_random_known_rank(self):
        # prepare the data:
        n1, n2, n3 = 100, 40, 20
        r1, r2, r3 = 5, 20, 10
        seed = 13
        rs = np.random.RandomState(seed)
        u1 = rs.randn(n1, r1)
        u2 = rs.randn(n2, r2)
        u3 = rs.randn(n3, r3)
        c = rs.randn(r1, r2, r3)
        inds, vals = dense_to_sparse_tensor(construct_tensor_from_tucker(u1, u2, u3, c))
        u0, u1, u2, core = hosvd(
            inds, vals, shape=(n1, n2, n3),
            rank=(r1, r2, r3), seed=0, verbose=False,
        )
        
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

    def test_random_known_rank_dense(self):
        # prepare the data:
        n1, n2, n3 = 100, 40, 20
        r1, r2, r3 = 5, 20, 10
        seed = 13
        rs = np.random.RandomState(seed)
        u1 = rs.randn(n1, r1)
        u2 = rs.randn(n2, r2)
        u3 = rs.randn(n3, r3)
        c = rs.randn(r1, r2, r3)
        x = construct_tensor_from_tucker(u1, u2, u3, c)
        inds, vals = dense_to_sparse_tensor(x)
        u0, u1, u2, core = hosvd_dense(
            x, rank=(r1, r2, r3), seed=0, verbose=False,
        )
        
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

    def test_random_known_rank_parallel(self):
        # prepare the data:
        n1, n2, n3 = 100, 40, 20
        r1, r2, r3 = 5, 20, 10
        seed = 13
        rs = np.random.RandomState(seed)
        u1 = rs.randn(n1, r1)
        u2 = rs.randn(n2, r2)
        u3 = rs.randn(n3, r3)
        c = rs.randn(r1, r2, r3)
        inds, vals = dense_to_sparse_tensor(construct_tensor_from_tucker(u1, u2, u3, c))
        u0, u1, u2, core = hosvd(
            inds, vals, shape=(n1, n2, n3),
            rank=(r1, r2, r3), seed=0,
            verbose=False, parallel=True
        )
        
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

class TestTucker2(unittest.TestCase):
    def test_random_known_rank(self):
        # prepare the data:
        n1, n2, n3 = 100, 40, 20
        r1, r2, r3 = 5, 20, 10
        seed = 13
        rs = np.random.RandomState(seed)
        u1 = rs.randn(n1, r1)
        u2 = rs.randn(n2, r2)
        u3 = rs.randn(n3, n3)
        c = rs.randn(r1, r2, n3)
        x = construct_tensor_from_tucker(u1, u2, u3, c)
        inds, vals = dense_to_sparse_tensor(x)
        _u0, _u1, _core = tucker2_dense(
            x, rank=(r1, r2, n3), identity_mode=2, 
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False
        )
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, _u0, _u1, np.eye(n3), _core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

