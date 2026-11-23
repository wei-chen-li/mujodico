"""Dense linear algebra for very small matrices (n <= SMALL_N) as fusible ops.

On GPU every cuSOLVER / cuBLAS call on a tiny matrix is a separate library
launch (plus pointer-setup kernels) that XLA can neither fuse nor batch with its
neighbours.  Inside a trajectory optimizer the dynamics (and the SAP contact
solve) run thousands of times per iteration, so these launches dominate the
time.  For n <= SMALL_N the Cholesky factorization, triangular solves and
products below are written as unrolled elementwise operations, which XLA fuses
into a few kernels; they compute the same quantities as the library routines
up to floating-point summation order.  Larger matrices use the library
routines.  Everything is differentiable and works under vmap.
"""

import jax
from jax import numpy as jp

SMALL_N = 16


def is_small(n: int) -> bool:
  return n <= SMALL_N


def mv(A, x):
  """A @ x for a matrix A and a vector x."""
  if is_small(A.shape[-1]):
    return jp.sum(A * x[None, :], axis=-1)
  return A @ x


def mtv(A, x):
  """A.T @ x."""
  if is_small(A.shape[-1]):
    return jp.sum(A * x[:, None], axis=0)
  return A.T @ x


def mm(A, B):
  """A @ B for 2-D A, B."""
  if is_small(max(A.shape[-1], B.shape[-1])):
    return jp.sum(A[:, :, None] * B[None, :, :], axis=1)
  return A @ B


def cho_factor(A):
  """Lower Cholesky factor of a symmetric positive-definite A, as
  ``(L, True)``; pass the result to :func:`cho_solve` (same contract as
  ``jax.scipy.linalg.cho_factor`` / ``cho_solve``).  A matrix that is not
  positive definite gives NaN entries, like the library routine."""
  n = A.shape[-1]
  if not is_small(n):
    return jax.scipy.linalg.cho_factor(A, lower=True)
  # Entries are built as a list of rows (no scatter updates).
  L = [[None] * n for _ in range(n)]
  for j in range(n):
    s = A[j, j]
    for k in range(j):
      s = s - L[j][k] * L[j][k]
    d = jp.sqrt(s)
    L[j][j] = d
    for i in range(j + 1, n):
      s = A[i, j]
      for k in range(j):
        s = s - L[i][k] * L[j][k]
      L[i][j] = s / d
  z = jp.zeros_like(A[0, 0])
  return (jp.stack([jp.stack([L[i][j] if j <= i else z for j in range(n)])
                    for i in range(n)]), True)


def cho_solve(c, b):
  """Solve ``A x = b`` given ``c = cho_factor(A)``; b is a vector or a
  matrix of right-hand sides (columns)."""
  L, lower = c
  n = L.shape[-1]
  if not is_small(n) or not lower:
    return jax.scipy.linalg.cho_solve(c, b)
  y = []
  for i in range(n):   # L y = b
    s = b[i]
    for k in range(i):
      s = s - L[i, k] * y[k]
    y.append(s / L[i, i])
  x = [None] * n
  for i in reversed(range(n)):   # L^T x = y
    s = y[i]
    for k in range(i + 1, n):
      s = s - L[k, i] * x[k]
    x[i] = s / L[i, i]
  return jp.stack(x)


@jax.custom_jvp
def spd_solve(A, b):
  """``A^{-1} b`` for a symmetric positive-definite A (vector or matrix b), by Cholesky.

  Same primal as ``cho_solve(cho_factor(A), b)``.  The derivative rule is
  ``dx = A^{-1} (db - dA x)`` with ``A^{-1}`` formed once from the factor and applied as a
  matrix product.  Differentiating through the Cholesky factorization instead
  produces batched triangular solves whose right-hand sides grow with the number of
  tangents (forward-mode Jacobians push all of them through at once), which is slow on
  the GPU for n > SMALL_N.  The two derivatives agree up to floating-point rounding."""
  return cho_solve(cho_factor(A), b)


@spd_solve.defjvp
def _spd_solve_jvp(primals, tangents):
  A, b = primals
  dA, db = tangents
  c = cho_factor(A)
  x = cho_solve(c, b)
  n = A.shape[-1]
  Ainv = cho_solve(c, jp.eye(n, dtype=A.dtype))
  return x, Ainv @ (db - dA @ x)

