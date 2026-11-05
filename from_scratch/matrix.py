"""Vector and matrix operations written out by hand. A matrix is a list of rows."""
import math
from operator import mul


def dot(a, b):
    return sum(map(mul, a, b))


def transpose(A):
    return [list(c) for c in zip(*A)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matvec(A, v):
    return [dot(r, v) for r in A]


def matmul(A, B):
    C = transpose(B)
    return [[dot(r, c) for c in C] for r in A]


def add_diag(A, d):
    """A + diag(d); d is a number or a list."""
    out = [r[:] for r in A]
    for i in range(len(A)):
        out[i][i] += d[i] if isinstance(d, list) else d
    return out


def gram(C, w=None):
    """XᵀX from the columns C of X, or XᵀWX when weights w are given."""
    W = C if w is None else [list(map(mul, c, w)) for c in C]
    n = len(C)
    G = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            G[i][j] = G[j][i] = dot(W[i], C[j])
    return G


def cholesky(A):
    """Lower-triangular L with A = LLᵀ, for a symmetric positive definite A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = A[i][j] - dot(Li[:j], Lj[:j])
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite")
                Li[j] = math.sqrt(s)
            else:
                Li[j] = s / Lj[j]
    return L


def forward(L, b):
    x = [0.0] * len(b)
    for i in range(len(b)):
        x[i] = (b[i] - dot(L[i][:i], x[:i])) / L[i][i]
    return x


def backward(L, y):
    """Solves Lᵀx = y."""
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def solve(A, b):
    L = cholesky(A)
    return backward(L, forward(L, b))


def inverse(A):
    L = cholesky(A)
    n = len(A)
    cols = [backward(L, forward(L, [1.0 if i == j else 0.0 for i in range(n)])) for j in range(n)]
    return transpose(cols)


def log_det(A):
    return 2 * sum(math.log(r[i]) for i, r in enumerate(cholesky(A)))


def eigh(A, vectors=True, sweeps=60, tol=1e-13):
    """Cyclic Jacobi rotations for a symmetric matrix.
    Returns eigenvalues from largest to smallest, and eigenvectors as a list of columns."""
    n = len(A)
    A = [r[:] for r in A]
    V = identity(n) if vectors else None
    scale = sum(A[i][i] ** 2 for i in range(n)) or 1.0
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off <= tol * tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if apq == 0.0:
                    continue
                theta = (A[q][q] - A[p][p]) / (2 * apq)
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                Ap, Aq = A[p], A[q]
                app, aqq = Ap[p], Aq[q]
                new_p = [c * x - s * y for x, y in zip(Ap, Aq)]
                new_q = [s * x + c * y for x, y in zip(Ap, Aq)]
                new_p[p] = app - t * apq
                new_q[q] = aqq + t * apq
                new_p[q] = new_q[p] = 0.0
                A[p], A[q] = new_p, new_q
                for k in range(n):
                    if k != p and k != q:
                        A[k][p], A[k][q] = new_p[k], new_q[k]
                if vectors:
                    for row in V:
                        x, y = row[p], row[q]
                        row[p], row[q] = c * x - s * y, s * x + c * y
    order = sorted(range(n), key=lambda i: -A[i][i])
    values = [A[i][i] for i in order]
    if not vectors:
        return values
    C = transpose(V)
    return values, [C[i] for i in order]


def mean(v):
    return sum(v) / len(v)


def variance(v, ddof=0):
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - ddof)


def standardise(train, *others):
    """Scales columns to mean 0 and variance 1 using the training rows only."""
    C = transpose(train)
    m = [mean(c) for c in C]
    s = [math.sqrt(variance(c)) for c in C]
    scale = lambda rows: [[(x - a) / b for x, a, b in zip(r, m, s)] for r in rows]
    return [scale(train)] + [scale(o) for o in others]
