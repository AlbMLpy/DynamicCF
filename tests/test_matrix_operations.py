import sys
import unittest

import numpy as np
from scipy.sparse import coo_matrix

sys.path.append('./')

from source.random_svd import random_svd, svd2mtx
from source.matrix_operations import (
    generate_matrix,
    get_svd_element,
    get_mf_element,
    sqrt_err_relative,
)

class TestMatrixOperations(unittest.TestCase):
    def setUp(self):
        self.rank = 2
        self.seed = 13
        self.a = generate_matrix((3, 4), self.rank, seed=self.seed)
        self.factors = random_svd(self.a, rank=self.rank, seed=self.seed)


    def test_generate_matrix(self):
        expected = self.rank
        actual = np.linalg.matrix_rank(self.a)
        self.assertEqual(actual, expected, msg="Ranks should be identical")

    def test_get_svd_element(self):
        i, j = 0, 1

        expected = svd2mtx(*self.factors)[i, j]
        actual = get_svd_element(i, j, *self.factors)
        self.assertAlmostEqual(actual, expected, msg="(0, 1) elements should be identical")
    
    def test_mf_element(self):
        i, j = 0, 1
        u, s, vt = self.factors
        vt = np.diag(s).dot(vt)

        expected = svd2mtx(*self.factors)[i, j]
        actual = get_mf_element(i, j, u, vt)
        self.assertAlmostEqual(actual, expected, msg="(0, 1) elements should be identical")

    def test_sqrt_err_relative_1ds(self):
        ca = coo_matrix(self.a)

        expected = 0
        actual = sqrt_err_relative(
            np.concatenate([ca.row[:, np.newaxis], ca.col[:, np.newaxis]], axis=1), ca.data, *self.factors
        )   
        self.assertAlmostEqual(actual, expected, msg='Difference should be 0')

    def test_sqrt_err_relative_2ds(self):
        ca = coo_matrix(self.a)
        u, s, vt = self.factors

        expected = 0
        actual = sqrt_err_relative(
            np.concatenate([ca.row[:, np.newaxis], ca.col[:, np.newaxis]], axis=1), 
            ca.data, 
            u, 
            np.diag(s),
            vt,

        )   
        self.assertAlmostEqual(actual, expected, msg='Difference should be 0')
