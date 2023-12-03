import sys
import unittest

import numpy as np
from scipy.sparse import coo_matrix

sys.path.append('./')
from source.tensor_operations import (
    kron_vec_jit,
    sparse_tensor_double_tensor,
    unfold_sparse,
    unfold_dense,
    fold_dense,
    construct_core,
    construct_core_parallel,
    construct_core_parallel_dense,
    tensordot2,
    tensordot,
    get_tucker_element,
    construct_tensor_from_tucker,
    dense_to_sparse_tensor,
    sparse_to_dense_tensor,
    D_TYPE
)

class TestTensorOperations(unittest.TestCase):

    def setUp(self):
        self.inds = np.array(
            [
                [0, 0, 0],
                [0, 3, 0],
                [2, 1, 0],
                [0, 2, 1],
                [1, 1, 1],
                [2, 0, 1],
                [0, 0, 2],
                [2, 1, 2],
                [0, 0, 3],
                [1, 2, 3],
                [2, 2, 3],
                [1, 2, 4],
                [2, 0, 4],
                [2, 3, 4],
            ]
        )

        self.vals = np.arange(1, 15)
        self.shape = (3, 4, 5)
        self.u0 = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0]
            ]
        )

        self.u1 = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0]
            ]
        )

        self.u2 = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0]
            ]
        )
        self.core = np.array(
            [
                [
                    [-0.24782446,  0.33595678],
                    [-0.12707375, -1.03521045]
                ],

                [
                    [ 0.63423559,  1.01342649],
                    [ 0.33298816, -1.06924143]
                ]
            ]
        )
        self.mode_0 = 0
        self.mode_1 = 1
        self.mode_2 = 2

        self.dense_tensor = np.zeros(shape=self.shape)
        for i, j in zip(self.inds, self.vals):
            i1, i2, i3 = i
            self.dense_tensor[i1, i2, i3] = j


    def test_unfold_sparse_0(self):
        expected = np.array(
            [
                [ 1.,  0.,  0.,  2.,  0.,  0.,  4.,  0.,  7.,  0.,  0.,  0.,  9., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 10.,  0.,  0.,  0., 12.,  0.],
                [ 0.,  3.,  0.,  0.,  6.,  0.,  0.,  0.,  0.,  8.,  0.,  0.,  0., 0., 11.,  0., 13.,  0.,  0., 14.]
            ]
        )
        actual = unfold_sparse(self.inds, self.vals, self.shape, self.mode_0).A
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_unfold_sparse_1(self):
        expected = np.array(
            [
                [ 1.,  0.,  0.,  0.,  0.,  6.,  7.,  0.,  0.,  9.,  0.,  0.,  0., 0., 13.],
                [ 0.,  0.,  3.,  0.,  5.,  0.,  0.,  0.,  8.,  0.,  0.,  0.,  0., 0.,  0.],
                [ 0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0., 10., 11.,  0., 12.,  0.],
                [ 2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 14.]
            ]
        )
        actual = unfold_sparse(self.inds, self.vals, self.shape, self.mode_1).A
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_unfold_sparse_2(self):
        expected = np.array(
            [
                [ 1.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  2.,  0.,  0.],
                [ 0.,  0.,  6.,  0.,  5.,  0.,  4.,  0.,  0.,  0.,  0.,  0.],
                [ 7.,  0.,  0.,  0.,  0.,  8.,  0.,  0.,  0.,  0.,  0.,  0.],
                [ 9.,  0.,  0.,  0.,  0.,  0.,  0., 10., 11.,  0.,  0.,  0.],
                [ 0.,  0., 13.,  0.,  0.,  0.,  0., 12.,  0.,  0.,  0., 14.]
            ]
        )
        actual = unfold_sparse(self.inds, self.vals, self.shape, self.mode_2).A
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_unfold_dense_0(self):
        mode = self.mode_0
        expected = unfold_sparse(self.inds, self.vals, self.shape, mode).A
        actual = unfold_dense(self.dense_tensor, mode)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_unfold_dense_1(self):
        mode = self.mode_1
        expected = unfold_sparse(self.inds, self.vals, self.shape, mode).A
        actual = unfold_dense(self.dense_tensor, mode)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_unfold_dense_2(self):
        mode = self.mode_2
        expected = unfold_sparse(self.inds, self.vals, self.shape, mode).A
        actual = unfold_dense(self.dense_tensor, mode)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_fold_dense_0(self):
        mode = self.mode_0
        expected = self.dense_tensor
        actual = fold_dense(unfold_dense(self.dense_tensor, mode), self.shape, mode)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_fold_dense_1(self):
        mode = self.mode_1
        expected = self.dense_tensor
        actual = fold_dense(unfold_dense(self.dense_tensor, mode), self.shape, mode)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_fold_dense_2(self):
        mode = self.mode_2
        expected = self.dense_tensor
        actual = fold_dense(unfold_dense(self.dense_tensor, mode), self.shape, mode)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_core_construction(self):
        expected = np.array(
            [
                [
                    [2359., 2359.],
                    [2359., 2359.]
                ],

                [
                    [2359., 2359.],
                    [2359., 2359.]
                ]
            ]
        )
        actual = construct_core(self.inds, self.vals, self.u0,  self.u1, self.u2)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_core_construction_parallel(self):
        expected = np.array(
            [
                [
                    [2359., 2359.],
                    [2359., 2359.]
                ],

                [
                    [2359., 2359.],
                    [2359., 2359.]
                ]
            ]
        )
        actual = construct_core_parallel(self.inds, self.vals, self.u0,  self.u1, self.u2)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_core_construction_parallel_dense(self):
        expected = np.array(
            [
                [
                    [2359., 2359.],
                    [2359., 2359.]
                ],

                [
                    [2359., 2359.],
                    [2359., 2359.]
                ]
            ]
        )
        x = sparse_to_dense_tensor(self.inds, self.vals, self.shape)
        actual = construct_core_parallel_dense(x, self.u0,  self.u1, self.u2)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_tensordot2_mode_01(self):
        expected = np.array(
            [
                [
                    [ 27.,  27.],
                    [ 27.,  27.]
                ],

                [
                    [ 50.,  50.],
                    [ 50.,  50.]
                ],

                [
                    [ 55.,  55.],
                    [ 55.,  55.]
                ],

                [
                    [168., 168.],
                    [168., 168.]
                ],

                [
                    [279., 279.],
                    [279., 279.]
                ]
            ]
        )
        actual = tensordot2(self.inds, self.vals, self.shape, self.u0, self.u1, (self.mode_0, self.mode_1))
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_tensordot2_mode_02(self):
        expected = np.array(
            [
                [
                    [289., 289.],
                    [289., 289.]
                ],

                [
                    [101., 101.],
                    [101., 101.]
                ],

                [
                    [340., 340.],
                    [340., 340.]
                ],

                [
                    [212., 212.],
                    [212., 212.]
                ]
            ]
        )
        actual = tensordot2(self.inds, self.vals, self.shape, self.u0, self.u2, (self.mode_0, self.mode_2))
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, "Difference should be 0")

    def test_tensordot2_mode_12(self):
        expected = np.array(
            [
                [
                    [ 90.,  90.],
                    [ 90.,  90.]
                ],

                [
                    [320., 320.],
                    [320., 320.]
                ],

                [
                    [543., 543.],
                    [543., 543.]
                ]
            ]
        )
        actual = tensordot2(self.inds, self.vals, self.shape, self.u1, self.u2, (self.mode_1, self.mode_2))
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0")

    def test_tensordot_mode_0(self):
        expected = np.array(
            [
                [1.,  9.,  0.,  2., 18., 10.,  4.,  0.,  7., 24.,  0.,  0.,  9., 0., 53.,  0., 39.,  0., 24., 42.],
                [ 1.,  9.,  0.,  2., 18., 10.,  4.,  0.,  7., 24.,  0.,  0.,  9., 0., 53.,  0., 39.,  0., 24., 42.]
            ]
        )
        actual = unfold_dense(tensordot(self.inds, self.vals, self.shape, self.u0.T, self.mode_0), self.mode_0)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0")

    def test_tensordot_mode_1(self):
        expected = np.array(
            [
                [ 9.,  0.,  6., 12., 10.,  6.,  7.,  0., 16.,  9., 30., 33., 0., 36., 69.],
                [ 9.,  0.,  6., 12., 10.,  6.,  7.,  0., 16.,  9., 30., 33., 0., 36., 69.]
            ]
        )
        actual = unfold_dense(tensordot(self.inds, self.vals, self.shape, self.u1.T, self.mode_1), self.mode_1)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0")

    def test_tensordot_mode_2(self):
        expected = np.array(
            [
                [ 58.,   0.,  77.,   0.,  10.,  27.,   8., 100.,  44.,   2.,   0., 70.],
                [ 58.,   0.,  77.,   0.,  10.,  27.,   8., 100.,  44.,   2.,   0., 70.]
            ]
        )
        actual = unfold_dense(tensordot(self.inds, self.vals, self.shape, self.u2.T, self.mode_2), self.mode_2)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0")
    
    def test_get_tucker_element(self):
        expected = -0.9764583966671614
        actual = get_tucker_element(0, 1, 2, self.u0, self.u1, self.u2, self.core)
        self.assertAlmostEqual(actual, expected, msg=f"Expected: {expected} != Actual: {actual}")

    def test_construct_tensor_from_tucker(self):
        expected = np.array(
            [
                [
                    [-0.16274307, -0.32548614, -0.48822921, -0.65097228, -0.81371535],
                    [-0.32548614, -0.65097228, -0.97645842, -1.30194456, -1.6274307],
                    [-0.48822921, -0.97645842, -1.46468763, -1.95291684, -2.44114605],
                    [-0.65097228, -1.30194456, -1.95291684, -2.60388912, -3.2548614]
                ],
                [
                    [-0.32548614, -0.65097228, -0.97645842, -1.30194456, -1.6274307],
                    [-0.65097228, -1.30194456, -1.95291684, -2.60388912, -3.2548614],
                    [-0.97645842, -1.95291684, -2.92937526, -3.90583368, -4.8822921],
                    [-1.30194456, -2.60388912, -3.90583368, -5.20777824, -6.5097228]
                ],
                [
                    [-0.48822921, -0.97645842, -1.46468763, -1.95291684, -2.44114605],
                    [-0.97645842, -1.95291684, -2.92937526, -3.90583368, -4.8822921],
                    [-1.46468763, -2.92937526, -4.39406289, -5.85875052, -7.32343815],
                    [-1.95291684, -3.90583368, -5.85875052, -7.81166736, -9.7645842]
                ]
            ]
        )
        actual = construct_tensor_from_tucker(self.u0, self.u1, self.u2, self.core)
        self.assertAlmostEqual(np.linalg.norm(actual[0] - expected[0]), 0, msg="Difference should be 0")

    def test_dense_to_sparse_tensor(self):
        expected = (
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]
                ]
            ),
            np.array(
                [-0.24782446,  0.33595678, -0.12707375, -1.03521045,  0.63423559, 1.01342649,  0.33298816, -1.06924143]
            )
        )
        actual = dense_to_sparse_tensor(self.core)
        self.assertAlmostEqual(np.linalg.norm(actual[0] - expected[0]), 0, msg="Difference should be 0")
        self.assertAlmostEqual(np.linalg.norm(actual[1] - expected[1]), 0, msg="Difference should be 0")

    def test_kron_vec_jit(self):
        a = np.arange(1, 3)
        b = np.arange(1, 5)
        
        expected = np.array([1., 2., 3., 4., 2., 4., 6., 8.])
        actual = kron_vec_jit(a, b)
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0")

    def test_sparse_tensor_double_tensor_mode_0(self):
        q = np.array(
            [
                [ 0.28032188,  0.90735399],
                [-0.67577357,  0.76568089],
                [-0.65675385,  1.04515706],
                [-0.72509302,  0.00263777]
            ],
            dtype=D_TYPE
        )
        
        expected = np.array(
            [
                [-159.95687075,  244.87467409],
                [-568.73554046,  870.66550787],
                [-965.07312021, 1477.41053366]
            ]
        )
        actual = sparse_tensor_double_tensor(
            self.inds, self.vals, self.shape, q=q, mode=self.mode_0, u1=self.u1, u2=self.u2
        )
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0, msg="Difference should be 0", places=5)



if __name__ == '__main__':
    unittest.main()
