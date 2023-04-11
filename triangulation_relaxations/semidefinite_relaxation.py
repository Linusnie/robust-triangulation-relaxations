import cvxpy
import numpy as np


class SDR:
    """Class for solving
    min tr(M @ X)
    s.t.    tr(A_i @ X) = 0
            tr(B_i @ x) <= 0
            tr(E @ X) = 1
    """

    def __init__(self, M, A, E, B=None):
        match M.shape:
            case (n, m) if n == m:
                self.n_variables = n
            case _:
                raise ValueError(f"M must be square, got shape {M.shape}")

        match A.shape:
            case (n, self.n_variables, self.n_variables):
                self.n_constraints = n
            case (self.n_variables, self.n_variables):
                A = A[None]
                self.n_constraints = 1
            case _:
                raise ValueError(f"Invalid shape for A: {A.shape}, for {self.n_variables=}")

        if B is not None:
            match B.shape:
                case (n, self.n_variables, self.n_variables):
                    self.n_inequalities = n
                case (self.n_variables, self.n_variables):
                    B = B[None]
                    self.n_inequalities = 1
                case _:
                    raise ValueError(f"Invalid shape for B: {B.shape}, for {self.n_variables=}")
        else:
            B = []
            self.n_inequalities = 0

        if E.shape != (self.n_variables, self.n_variables):
            raise ValueError(f"Invalid shape for E: {E.shape}, for {self.n_variables=}")

        self.M = M
        self.A = A
        self.E = E
        self.B = B

        self.X = cvxpy.Variable((self.n_variables, self.n_variables), PSD=True)
        self.problem = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.trace(self.X @ self.M)),
            [cvxpy.trace(self.X @ Ai) == 0 for Ai in A] +
            [cvxpy.trace(self.X @ Bi) <= 0 for Bi in B] +
            [self.X >> 0, cvxpy.trace(self.X @ self.E) == 1]
        )
        self.solved = False

    def set_parameters(self, M, A, E, B=None):
        self.M.value = M
        for i in range(self.n_constraints):
            self.A[i].value = A[i]
        self.E.value = E
        if B is not None:
            for i in range(self.n_inequalities):
                self.B[i].value = B[i]

    @property
    def dual(self):
        lmbd = cvxpy.Variable()
        mu = cvxpy.Variable(len(self.A))
        S = self.M - self.E * lmbd
        for i in range(len(self.A)):
            S -= self.A[i] * mu[i]
        dual_problem = cvxpy.Problem(
            cvxpy.Maximize(lmbd),
            [S >> 0]
        )
        return dual_problem, mu, lmbd, S

    def assert_solved(self):
        if not self.solved:
            raise ValueError("Problem as not been solved")

    def solve(self, **kwargs):
        result = self.problem.solve(**kwargs)
        self.solved = True
        return result

    def get_rank1(self):
        self.assert_solved()
        u, s, vt = np.linalg.svd(self.X.value)
        x = vt[0] * s[0]
        x /= np.sqrt(x @ self.E @ x)
        return x

    def constraints(self, X=None):
        if X is None:
            X = self.X.value
        if X.shape == (self.n_variables,):
            X = np.outer(X, X)
        return [np.trace(X @ Ai) for Ai in self.A] + [np.trace(X @ self.E)]

    def objective(self, X=None):
        if X is None:
            X = self.X.value
        if X.shape == (self.n_variables,):
            X = np.outer(X, X)
        return np.trace(X @ self.M)

    def get_corank_1_multipliers(self):
        raise NotImplementedError()


def project_psd(Z):
    l, v = np.linalg.eigh(Z)
    return v @ np.diag(np.clip(l, 0, None)) @ v.T


def one_hot(n, i):
    out = np.zeros(n)
    out[i] = 1
    return out


def one_hot_matrix(n, i, j):
    out = np.zeros((n, n))
    out[i, j] = 1
    return out
