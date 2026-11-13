"""
Stiefel manifold optimization, Orthogonal Procrustes alignment, and Gram-Schmidt
orthogonalization for narrative embedding matrices.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row of a 2D array to unit L2 norm."""
    arr = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.where(norms > eps, arr / norms, arr)


def gram_schmidt(N_matrix: np.ndarray) -> np.ndarray:
    """Sequential Gram-Schmidt row-wise orthogonalization."""
    n, dim = N_matrix.shape
    V = np.zeros_like(N_matrix, dtype=float)

    for i in range(n):
        v = N_matrix[i].copy().astype(float)
        for j in range(i):
            v -= np.dot(v, V[j]) * V[j]
        norm = np.linalg.norm(v)
        V[i] = v / norm if norm > 1e-10 else v

    return V


def stiefel_optimize(
    N_matrix: np.ndarray,
    softmin_temp: float = 0.05,
    max_iter: int = 300,
    verbosity: int = 1,
):
    """
    Optimization on the Stiefel manifold St(dim, n).
    Maximizes min_i <N_i, V_i> subject to V @ V.T = I_n using a smooth soft-min
    surrogate and Riemannian Conjugate Gradient initialized with Gram-Schmidt.
    """
    n, dim = N_matrix.shape
    manifold = Stiefel(dim, n)

    @pymanopt.function.autograd(manifold)
    def cost(V):
        Vt = V.T
        projections = anp.sum(N_matrix * Vt, axis=1)
        softmin = -softmin_temp * anp.log(anp.sum(anp.exp(-projections / softmin_temp)))
        return -softmin

    problem = pymanopt.Problem(manifold, cost)
    optimizer = ConjugateGradient(verbosity=verbosity, max_iterations=max_iter)

    V_init = gram_schmidt(N_matrix).T
    result = optimizer.run(problem, initial_point=V_init)

    return result.point.T, result


def procrustes_stiefel(N_matrix: np.ndarray) -> np.ndarray:
    """
    Closed-form solution to the Orthogonal Procrustes problem on St(dim, n).
    Maximizes sum_i <N_i, V_i> (= minimizes ||N - V||_F^2) subject to V @ V.T = I_n.
    """
    N = N_matrix.T
    U, _, Wt = np.linalg.svd(N, full_matrices=False)
    V = U @ Wt
    return V.T


def decompose_narratives(
    N_matrix_unorm: np.ndarray,
    softmin_temp: float = 0.05,
    max_iter: int = 300,
    verbosity: int = 1,
):
    """
    Given unnormalized narrative embeddings (n, dim), returns:
      - N_norm: row-normalized original narratives
      - V_orth: Stiefel-orthogonalized pure narratives
      - R_norm: row-normalized residual vectors (N_matrix_unorm - V_orth)
      - result: pymanopt optimization result
    """
    if hasattr(N_matrix_unorm, "to_numpy"):
        N_unorm = np.vstack(N_matrix_unorm.to_numpy()).astype(float)
    else:
        N_unorm = np.asarray(N_matrix_unorm, dtype=float)
    N_norm = normalize_rows(N_unorm)
    V_orth, result = stiefel_optimize(
        N_norm, softmin_temp=softmin_temp, max_iter=max_iter, verbosity=verbosity
    )
    R_norm = normalize_rows(N_unorm - V_orth)
    return N_norm, V_orth, R_norm, result
