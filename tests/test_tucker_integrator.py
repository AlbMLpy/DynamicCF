import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append('./')

from source.rp_hooi import rp_hooi
from source.tensor_operations import (
    sqrt_err_relative,
    construct_tensor_from_tucker,
    dense_to_sparse_tensor,
    unfold_sparse,
)
from source.tucker_integrator import (
    tucker_intergrator, 
    update_tucker_new_vectors,
    update_tucker_new_matrix,
)

class TestTuckerIntegrator(unittest.TestCase):
    def setUp(self):
        self.shape = 100, 40, 20
        self.rank = 5, 20, 10
        self.seed = 13
        rs = np.random.RandomState(self.seed)
        n1, n2, n3 = self.shape
        r1, r2, r3 = self.rank
        u1 = rs.randn(n1, r1)
        u2 = rs.randn(n2, r2)
        u3 = rs.randn(n3, r3)
        c = rs.randn(r1, r2, r3)
        self.inds, self.vals = dense_to_sparse_tensor(
            construct_tensor_from_tucker(u1, u2, u3, c)
        )
        *self.factors, self.core = rp_hooi(
            self.inds, self.vals, shape=(n1, n2, n3),
            rank=(r1, r2, r3), n_iter=25, growth_tol=0.01,
            n_power_iter=0, oversampling_factor=10,
            seed=self.seed, verbose=False,
        )

    def test_random_known_rank(self):
        # prepare the data:
        new_inds = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1]
            ]
        )
        new_vals = np.ones(4)
        u0, u1, u2, core = tucker_intergrator(
            self.factors, self.core, new_inds, new_vals, shape=self.shape, parallel=True
        )
        inds = np.concatenate([self.inds, new_inds], axis=0)
        vals = np.concatenate([self.vals, new_vals], axis=0)
        
        expected = 1
        actual = sqrt_err_relative(inds, vals, u0, u1, u2, core)
        self.assertLessEqual(actual, expected, msg="Difference should be 0")

    def test_update_tucker_new_vectors_shapes_mode_0(self):
        mode = 0
        new_entities = unfold_sparse(self.inds, self.vals, self.shape, mode)
        hm = new_entities.shape[0] // 5
        *nf, _ = update_tucker_new_vectors(
            self.factors, 
            self.core, 
            new_entities.tocsr()[[i for i in range(hm)]], 
            mode, 
            seed=self.seed
        )

        expected = (self.factors[mode].shape[0] + hm, self.factors[mode].shape[1])
        actual = nf[mode].shape 
        self.assertEqual(actual, expected, msg=f"factor shape = {actual} != expected shape = {expected}")

    def test_update_tucker_new_vectors_shapes_mode_1(self):
        mode = 1
        new_entities = unfold_sparse(self.inds, self.vals, self.shape, mode)
        hm = new_entities.shape[0] // 5
        *nf, _ = update_tucker_new_vectors(
            self.factors, 
            self.core, 
            new_entities.tocsr()[[i for i in range(hm)]], 
            mode, 
            seed=self.seed
        )

        expected = (self.factors[mode].shape[0] + hm, self.factors[mode].shape[1])
        actual = nf[mode].shape 
        self.assertEqual(actual, expected, msg=f"factor shape = {actual} != expected shape = {expected}")

    def test_update_tucker_new_vectors_1i(self):
        mode = 1
        new_entities = unfold_sparse(
            self.inds, self.vals, self.shape, mode
        )
        *nf, _ = update_tucker_new_vectors(
            self.factors, 
            self.core, 
            new_entities.tocsr()[[0]], 
            mode, 
            seed=self.seed
        )

        expected = (
            self.factors[mode].shape[0] + 1, 
            self.factors[mode].shape[1]
        )
        actual = nf[mode].shape 
        self.assertEqual(actual, expected, msg=f"factor shape = {actual} != expected shape = {expected}")

    def test_update_tucker_new_vectors_1u(self):
        mode = 0
        new_entities = unfold_sparse(
            self.inds, self.vals, self.shape, mode
        )
        *nf, _ = update_tucker_new_vectors(
            self.factors, 
            self.core, 
            new_entities.tocsr()[[0]], 
            mode, 
            seed=self.seed
        )

        expected = (
            self.factors[mode].shape[0] + 1, 
            self.factors[mode].shape[1]
        )
        actual = nf[mode].shape 
        self.assertEqual(actual, expected, msg=f"factor shape = {actual} != expected shape = {expected}")

    def test_update_tucker_new_matrix_shapes(self):
        new_data = pd.DataFrame(
            {
                0: [0, 0, 1, 2, 1, 2, 3, 4, 5, 6],
                1: [0, 1, 0, 2, 1, 0, 1, 3, 3, 4],
                'relevance': [1]*10,
                'order': [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            }
        )
        new_shape = (7, 5, self.core.shape[2])
        new_inds = new_data[[0, 1, 'order']].to_numpy()
        new_vals = new_data['relevance'].values
        *nf, _ = update_tucker_new_matrix(
            self.factors, self.core, new_inds, new_vals, new_shape, seed=self.seed
        )
        for mode in range(2):
            actual = nf[mode].shape
            expected = (self.factors[mode].shape[0] + new_shape[mode], self.factors[mode].shape[1])
            self.assertEqual(actual, expected, msg=f"factor shape = {actual} != expected shape = {expected}")

    def test_update_tucker_new_matrix_1u_1i(self):
        new_data = pd.DataFrame(
            {
                0: [0,],
                1: [0,],
                'relevance': [1,],
                'order': [0,],
            }
        )
        new_shape = (1, 1, self.core.shape[2])
        new_inds = new_data[[0, 1, 'order']].to_numpy()
        new_vals = new_data['relevance'].values
        *nf, _ = update_tucker_new_matrix(
            self.factors, self.core, new_inds, new_vals, new_shape, seed=self.seed
        )
        for mode in range(2):
            actual = nf[mode].shape
            expected = (self.factors[mode].shape[0] + new_shape[mode], self.factors[mode].shape[1])
            self.assertEqual(actual, expected, msg=f"factor shape = {actual} != expected shape = {expected}")
