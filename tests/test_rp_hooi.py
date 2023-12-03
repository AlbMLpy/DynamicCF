import sys
import unittest

import numpy as np

sys.path.append('./')

from source.rp_hooi import rp_hooi, ga_satf, tucker2


from source.tensor_operations import (
    gen_hilbert_tensor,
    sqrt_err_relative,
    construct_tensor_from_tucker,
    dense_to_sparse_tensor,
)

class TestHooi(unittest.TestCase):
    def test_bad_rank(self):
        with self.assertRaises(RuntimeError):
            bad_rank = (5, 2, 2)

            shape = (20, 20, 20)
            inds, vals = gen_hilbert_tensor(shape)
            rp_hooi(
                inds, vals, shape=shape,
                rank=bad_rank, n_iter=25, growth_tol=0.01,
                n_power_iter=0, oversampling_factor=10,
                seed=0, verbose=False,
            )

    def test_hilbert_tensor(self):
        # prepare the data:
        shape = (20, 20, 20)
        inds, vals = gen_hilbert_tensor(shape)
        r = 9
        rank = (r, r, r)
        u0, u1, u2, core = rp_hooi(
            inds, vals, shape=shape,
            rank=rank, n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False,
        )
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

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
        u0, u1, u2, core = rp_hooi(
            inds, vals, shape=(n1, n2, n3),
            rank=(r1, r2, r3), n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False,
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
        u0, u1, u2, core = rp_hooi(
            inds, vals, shape=(n1, n2, n3),
            rank=(r1, r2, r3), n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False, parallel=True
        )
        
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

class TestTucker2(unittest.TestCase):
    def test_random_known_rank_parallel(self):
        # prepare the data:
        n1, n2, n3 = 100, 40, 20
        r1, r2, r3 = 5, 20, 10
        seed = 13
        rs = np.random.RandomState(seed)
        u1 = rs.randn(n1, r1)
        u2 = rs.randn(n2, r2)
        u3 = rs.randn(n3, n3)
        c = rs.randn(r1, r2, n3)
        inds, vals = dense_to_sparse_tensor(construct_tensor_from_tucker(u1, u2, u3, c))
        _u0, _u1, _core = tucker2(
            inds, vals, shape=(n1, n2, n3),
            rank=(r1, r2, n3), identity_mode=2, 
            n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False, parallel=True
        )
        expected = 0.0
        actual = sqrt_err_relative(inds, vals, _u0, _u1, np.eye(n3), _core)
        self.assertAlmostEqual(actual - expected, 0, msg="Difference should be 0")

class TestGASATF(unittest.TestCase):
    def test_gasatf_equal_rp_hooi(self):
        # prepare the data:
        n_pos = 20
        shape = (20, 20, n_pos)
        inds, vals = gen_hilbert_tensor(shape)
        r = 9
        rank = (r, r, r)
        rp_hooi_params = rp_hooi(
            inds, vals, shape=shape,
            rank=rank, n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False,
        )
        ga_satf_params = ga_satf(
            inds, vals, shape=shape,
            rank=rank, attention_mtx=np.eye(n_pos), 
            n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=0, verbose=False,
        )
        for expected, actual in zip(rp_hooi_params, ga_satf_params):
            self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0")
