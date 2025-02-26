import scipy
import numpy as np

from triangulation_relaxations import so3, se3, geometry
from triangulation_relaxations.se3 import Se3
from triangulation_relaxations.semidefinite_relaxation import SDR, one_hot, one_hot_matrix


class TriangulationProblem:
    def __init__(self, poses: Se3, point: np.array, K: np.array):
        self.poses = poses
        self.n_poses = len(poses)
        self.point = point
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def get_observations(self, sigma=0):
        x = geometry.reproject(self.point, self.poses, self.K)
        x += np.random.randn(*x.shape) * sigma
        return x

    def get_transformed(self):
        return geometry.transform(self.poses.inverse(), self.point, self.K_inv)

    @property
    def F(self):
        return geometry.get_fundamental(se3.get_relative(self.poses), self.K_inv)

    @property
    def E(self):
        return geometry.get_essential(se3.get_relative(self.poses))

    def __str__(self):
        return f"Triangulation problem\n" \
               f"n_poses: {len(self.poses)}\n" \
               f"point: {self.point}\n" \
               f"K: {self.K}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return TriangulationProblem(
            poses=self.poses[item],
            point=self.point,
            K=self.K,
        )


class TriangulationSDR(SDR):
    def __init__(self, poses: Se3, observations: np.array, K: np.array, scale=1., c=None):
        self.n_poses = len(poses)
        self.poses = poses
        self.observations = observations
        self.scale = scale
        self.K = scale_intrinsics(K, scale)
        self.K_inv = np.linalg.inv(self.K)

        M = get_l2_matrix(observations.ravel() * scale)
        A = self.get_epipolar_constraints(self.poses, self.K_inv)
        super(TriangulationSDR, self).__init__(M, A, one_hot_matrix(2 * self.n_poses + 1, -1, -1))

    def get_solution(self, eps: float = 1e-8):
        self.assert_solved()
        _, s_sdr, vt = np.linalg.svd(self.X.value)
        z = vt[0]
        estimated_point, s_algebraic = triangulate_algebraic(
            (z[:-1] / z[-1]).reshape((self.n_poses, 2)),
            self.poses,
            K=np.linalg.inv(self.K_inv)
        )
        # check that solution is rank 1 and that the triangulated point is unique
        success = (s_sdr[1] < eps) and (s_algebraic[-1] < eps) and (s_algebraic[-2] > eps)
        return {
            'estimated_point': estimated_point,
            'z': z,
            's_algebraic': s_algebraic,
            **{f's{i}': s_sdr[i] for i in range(len(s_sdr))},
            'success': success,
        }

    @staticmethod
    def get_epipolar_constraints(poses: Se3, K_inv: np.ndarray):
        n_poses = len(poses)
        fundamental_matrices = geometry.get_relative_fundamental(poses, K_inv)
        A = []
        for i in range(n_poses):
            for j in range(i + 1, n_poses):
                A.append(get_epipolar_constraint(fundamental_matrices[i, j], n_poses, i, j))
        return np.array(A)

    def z_from_point(self, point):
        return geometry.homogenize(geometry.reproject(point, self.poses, np.linalg.inv(self.K_inv)).ravel())

    def set_observations(self, observations):
        self.observations = observations
        self.M = get_l2_matrix(observations.ravel())
        self.solved = False

    def get_corank_1_multipliers(self):
        return np.zeros(self.n_constraints)


class RobustTriangulationSDR(SDR):
    def __init__(self, poses: Se3, observations: np.array, K: np.array, c: np.array = None, scale=1.,
                 with_inequalities=True):
        n_poses = len(poses)
        self.n_poses = n_poses
        self.poses = poses
        self.observations = observations
        self.scale = scale
        self.K = scale_intrinsics(K, self.scale)
        self.K_inv = np.linalg.inv(self.K)
        if c is None:
            c = np.ones(self.n_poses)
        self.c = c * (self.scale ** 2)

        M = get_robust_l2_matrix(observations * self.scale, self.c)
        fundamental_matrices = geometry.get_relative_fundamental(poses, self.K_inv)

        n_variables = 3 * n_poses + 1
        A = np.zeros((n_poses * (n_poses - 1) // 2 + 3 * n_poses, n_variables, n_variables))
        constraint_index = 0

        # xi @ Fij @ xj = 0
        for i in range(n_poses):
            for j in range(i + 1, n_poses):
                indices = np.ix_(
                    [constraint_index],
                    [3 * i, 3 * i + 1, 3 * i + 2],
                    [3 * j, 3 * j + 1, 3 * j + 2]
                )
                A[indices] = fundamental_matrices[i, j]
                constraint_index += 1

        # thetai * xi = xi
        for i in range(n_poses):
            A[constraint_index, 3 * i, 3 * i + 2] = 1
            A[constraint_index, 3 * i, -1] = -1
            constraint_index += 1

            A[constraint_index, 3 * i + 1, 3 * i + 2] = 1
            A[constraint_index, 3 * i + 1, -1] = -1
            constraint_index += 1

            A[constraint_index, 3 * i + 2, 3 * i + 2] = 1
            A[constraint_index, 3 * i + 2, -1] = -1
            constraint_index += 1

        A = (A + A.transpose(0, 2, 1)) / 2

        if with_inequalities:
            # at least 2 inliers
            B = np.zeros((n_variables, n_variables))
            for i in range(n_poses):
                B[3 * i + 2, 3 * i + 2] = -1
            B[-1, -1] = 2
        else:
            B = None

        super(RobustTriangulationSDR, self).__init__(M, A, one_hot_matrix(3 * n_poses + 1, -1, -1), B)

    def get_solution(self, eps: float = 1e-8):
        self.assert_solved()
        _, s_sdr, vt = np.linalg.svd(self.X.value)
        z = vt[0]
        points = z[:-1].reshape((self.n_poses, 3)) / z[-1]
        inlier_mask = np.round(points[:, -1]).astype(bool)

        K = np.array([self.K for _ in range(self.n_poses)]) if self.K.ndim == 2 else self.K
        estimated_point, s_algebraic = triangulate_algebraic(points[inlier_mask, :2], self.poses[inlier_mask],
                                                             K=K[inlier_mask])

        # check that solution is rank 1 and that the estimated view rays all intersect at a single point
        # (second condition can fail for coplanar poses where the inlier observations are almost on the epipolar plane)
        success = (s_sdr[1] < eps) and (s_algebraic[-1] < eps) and (s_algebraic[-2] > eps)
        return {
            'estimated_point': estimated_point,
            'estimated_inlier_mask': inlier_mask,
            'z': z,
            's_algebraic': s_algebraic,
            **{f's{i}': s_sdr[i] for i in range(len(s_sdr))},
            'success': success,
        }

    def get_corank_1_multipliers(self):
        mu = np.zeros(self.n_constraints)
        mu[self.n_poses * (self.n_poses - 1) // 2:] = -np.kron(self.c, [0, 0, 1])
        return mu


class TriangulationFractionalSDR(SDR):
    def __init__(self, poses: Se3, observations: np.array, K: np.array, scale=1., c=None, full_constraints=True):
        self.n_poses = len(poses)
        self.poses = poses
        self.observations = observations
        self.scale = scale
        self.K = scale_intrinsics(K, self.scale)
        self.K_inv = np.linalg.inv(self.K)
        M = np.kron(get_l2_matrix(self.observations.ravel() * self.scale, one_first=False), np.eye(4))
        A = np.vstack([
            self.get_reprojection_constraints(poses, self.K, full_constraints=full_constraints),
            self.get_kronecker_constraints(2 * self.n_poses + 1, 4)
        ])
        E = scipy.linalg.block_diag(np.zeros((self.n_poses * 8, self.n_poses * 8)), np.eye(4))
        super(TriangulationFractionalSDR, self).__init__(M, A, E)

    def get_solution(self, eps: float = 1e-8):
        self.assert_solved()
        _, s, vt = np.linalg.svd(self.X.value)
        z = vt[0]
        z /= np.linalg.norm(z[-4:])
        v, estimated_point = self.decompose_kronecker(z, 2 * self.n_poses + 1, 4)
        estimated_point = estimated_point[:-1] / estimated_point[-1]  # TODO: handle points at infinity
        success = s[1] < eps
        return {
            'estimated_point': estimated_point,
            'z': z,
            **{f's{i}': s[i] for i in range(len(s))},
            'success': success,
        }

    @staticmethod
    def get_a_b(poses, K):
        inv_poses = np.einsum('...ij, ...jk -> ...ik', K, poses.inverse().T[:, :-1])
        n_poses = len(poses)
        a = np.vstack([inv_poses[i, :2] for i in range(n_poses)])
        b = np.vstack([np.array([inv_poses[i, 2], inv_poses[i, 2]]) for i in range(len(poses))]).reshape((-1, 4))
        return a, b

    @staticmethod
    def get_reprojection_constraints(poses, K, full_constraints=True):
        a, b = TriangulationFractionalSDR.get_a_b(poses, K)
        n_poses = len(poses)
        A = []
        for i in range(len(a)):
            for j in range(n_poses * 8 + 4) if full_constraints else range(4):
                # v_j z^Tb_i v_i = v_j z^Ta_i
                Ai = np.outer(
                    np.kron(one_hot(2 * n_poses + 1, i), b[i]) - np.kron(one_hot(2 * n_poses + 1, -1), a[i]),
                    one_hot(8 * n_poses + 4, -j - 1)
                )
                A.append((Ai + Ai.T) * .5)
        return np.array(A)

    @staticmethod
    def get_kronecker_constraints(m, n):
        """Get constraints corresponding to x = kron(a, b) with len(a)=m, len(b)=n"""
        constraints = []
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                for j1 in range(n):
                    for j2 in range(j1 + 1, n):
                        A = np.zeros((n * m, n * m))
                        A[i1 * n + j1, i2 * n + j2] = 1
                        A[i1 * n + j2, i2 * n + j1] = -1
                        constraints.append((A + A.T) * .5)
        return np.array(constraints)

    @staticmethod
    def decompose_kronecker(x, m, n):
        """Decompose x as x = kron(a, b) with len(a)=m, len(b)=n and a[-1]=1. Assuming the decomposition exists."""
        u, s, vt = np.linalg.svd(x.reshape((m, n)))
        a = u[:, 0] / u[-1, 0]
        b = vt[0] * s[0] * u[-1, 0]
        return a, b

    def get_corank_1_multipliers(self):
        # M[4:, 4:] = I + kron_constraints(alpha), so it's psd as long as alpha << 1 by gershgorin circle theorem
        mu = np.zeros(self.n_constraints)
        mu[4 * 2 * self.n_poses:] = np.random.randn(self.n_constraints - 4 * 2 * self.n_poses) * 0.001
        return mu


class RobustTriangulationFractionalSDR(SDR):
    def __init__(self, poses: Se3, observations: np.array, K: np.array, c: np.array = None, scale=1.,
                 full_constraints=True, with_inequalities=True):
        # TODO: handle K
        self.n_poses = len(poses)
        self.n_variables = 12 * self.n_poses + 4
        self.poses = poses
        self.observations = observations
        self.scale = scale
        self.K = scale_intrinsics(K, self.scale)
        self.K_inv = np.linalg.inv(self.K)
        if c is None:
            c = np.ones(self.n_poses)
        self.c = c * (self.scale ** 2)

        M = np.kron(get_robust_l2_matrix(observations * self.scale, self.c), np.eye(4))
        A = np.vstack([
            self.get_reprojection_constraints(poses, self.K, full_constraints),
            self.get_theta_constraints(),
            self.get_kronecker_constraints(3 * self.n_poses + 1, 4),
        ])
        E = scipy.linalg.block_diag(np.zeros((self.n_poses * 12, self.n_poses * 12)), np.eye(4))

        # At least 2 inliers
        # sum((theta_i * X_k)**2) >= 2X_k**2
        if with_inequalities:
            B = []
            for k in range(4):
                Bk = np.zeros((self.n_variables, self.n_variables))
                for i in range(self.n_poses):
                    Bk[4 * (3 * i + 2) + k, 4 * (3 * i + 2) + k] = -1
                    Bk[-4:, -4:][k, k] = 2
                B.append(Bk)
            B = np.array(B)
        else:
            B = None
        super(RobustTriangulationFractionalSDR, self).__init__(M, A, E, B)

    def get_solution(self, eps: float = 1e-8):
        self.assert_solved()
        _, s, vt = np.linalg.svd(self.X.value)
        z = vt[0]
        z /= np.linalg.norm(z[-4:])
        v, estimated_point = self.decompose_kronecker(z, 3 * self.n_poses + 1, 4)
        inlier_mask = np.round(v[:-1].reshape(self.n_poses, 3)[:, -1]).astype(bool)
        estimated_point = estimated_point[:-1] / estimated_point[-1]  # TODO: handle points at infinity
        success = s[1] < eps
        return {
            'estimated_point': estimated_point,
            'estimated_inlier_mask': inlier_mask,
            'z': z,
            **{f's{i}': s[i] for i in range(len(s))},
            'success': success
        }

    def get_theta_constraints(self):
        A = []
        for i in range(self.n_poses):
            for k in range(4):
                for l in range(4):
                    for d in range(3):
                        # (z_k x_i)(z_l theta_i) = (z_k)(z_l x_i)
                        Ai = np.zeros((self.n_variables, self.n_variables))
                        Ai[4 * (3 * i + d) + k, 4 * (3 * i + 2) + l] = 1
                        Ai[12 * self.n_poses + k, 4 * (3 * i + d) + l] = -1
                        A.append((Ai + Ai.T) * .5)
        return np.array(A)

    @staticmethod
    def get_reprojection_constraints(poses, K, full_constraints=True):
        inv_poses = np.einsum('...ij, ...jk -> ...ik', K, poses.inverse().T[:, :-1])
        n_poses = len(poses)
        n_variables = 12 * n_poses + 4
        A = []
        for i in range(n_poses):
            for d in range(2):
                for j in range(n_variables) if full_constraints else [-4, -3, -2, -1]:
                    b = inv_poses[i, 2]
                    Ai = np.zeros((n_variables, n_variables))
                    Ai[j, 4 * (3 * i + 2):4 * (3 * i + 3)] = -inv_poses[i, d]
                    Ai[j, 4 * (3 * i + d):4 * (3 * i + d + 1)] = b
                    A.append((Ai + Ai.T) * .5)
        return np.array(A)

    @staticmethod
    def get_kronecker_constraints(m, n):
        """Get constraints corresponding to x = kron(a, b) with len(a)=m, len(b)=n"""
        constraints = []
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                for j1 in range(n):
                    for j2 in range(j1 + 1, n):
                        A = np.zeros((n * m, n * m))
                        A[i1 * n + j1, i2 * n + j2] = 1
                        A[i1 * n + j2, i2 * n + j1] = -1
                        constraints.append((A + A.T) * .5)
        return np.array(constraints)

    @staticmethod
    def decompose_kronecker(x, m, n):
        """Decompose x as x = kron(a, b) with len(a)=m, len(b)=n and a[-1]=1. Assuming the decomposition exists."""
        u, s, vt = np.linalg.svd(x.reshape((m, n)))
        a = u[:, 0] / u[-1, 0]
        b = vt[0] * s[0] * u[0, 0]
        return a, b

    def objective(self, X=None):
        return super(RobustTriangulationFractionalSDR, self).objective(X)


def triangulation_residuals(point, poses, observations, K):
    point_reprojections = geometry.reproject(point, poses, K=K)
    residuals = (point_reprojections - observations).reshape(-1)
    return residuals


def triangulate_nonlinear(poses: Se3, observations: np.array, K, initial_point):
    result = scipy.optimize.least_squares(
        triangulation_residuals, initial_point,
        args=(poses, observations, K), gtol=1e-12, ftol=1e-12, xtol=1e-12
    )
    return result.x, result


def robust_cost(point, poses, observations, intrinsics, c):
    square_norms = np.linalg.norm(geometry.reproject(point, poses, intrinsics) - observations, axis=1) ** 2
    outlier_mask = square_norms > c
    if outlier_mask.any():
        square_norms[outlier_mask] = c[outlier_mask]
    return square_norms.sum()


def get_epipolar_constraint(F, n, i, j):
    F_constraint = np.zeros((2 * n + 1, 2 * n + 1))
    F_constraint[np.ix_([2 * i, 2 * i + 1, -1], [2 * j, 2 * j + 1, -1])] = F
    F_constraint += F_constraint.T
    return F_constraint


def get_aholt_problem(n_poses, mode="sphere", angle_jitter_deg=0., K=None, radius=2., hw=None):
    """Sphere experiment from Aholt et al"""

    def get_problem():
        if mode == "sphere":
            t = np.random.randn(n_poses, 3) - 0.5
        elif mode == "circle":
            t = np.zeros((n_poses, 3))
            t[:, :2] = np.random.randn(n_poses, 2)
        t *= radius / np.linalg.norm(t, axis=1)[:, None]
        R = so3.get_rotaitons_facing_point(np.zeros(3), t)
        if angle_jitter_deg > 0:
            R = np.einsum('...ij, ...jk->...ik', R,
                          so3.rotvec_to_r(np.random.randn(n_poses, 3) * np.pi * angle_jitter_deg / 180))
        return TriangulationProblem(
            poses=Se3(R, t),
            point=np.random.rand(3),
            K=K if K is not None else np.eye(3)
        )

    problem = get_problem()
    if hw is not None:
        h, w = hw
        all_visible = False
        i = 0
        while not all_visible:
            problem = get_problem()
            observations = problem.get_observations(0.)
            all_visible = np.logical_and.reduce((
                observations[:, 0] > 0, observations[:, 0] < h,
                observations[:, 1] > 0, observations[:, 1] < w,
            )).all()
            i += 1
            if i > 10000:
                raise ValueError("too many tries")
    return problem


def get_sphere_problem(n_poses, angle_jitter_deg=5, radius=2, mode="sphere", point=None):
    if mode == "sphere":
        t = np.random.randn(n_poses, 3)
    elif mode == "circle":
        t = np.zeros((n_poses, 3))
        t[:, :2] = np.random.randn(n_poses, 2)
    t *= radius / np.linalg.norm(t, axis=1)[:, None]

    if point is None:
        point = np.random.rand(3)

    R = so3.get_rotaitons_facing_point(point, t)
    R = np.einsum('...ij, ...jk->...ik', R,
                  so3.rotvec_to_r(np.random.randn(n_poses, 3) * np.pi * angle_jitter_deg / 180))
    return TriangulationProblem(
        poses=Se3(R, t),
        point=point,
        K=np.eye(3)
    )


def triangulate_2view(points, poses, intrinsics):
    proj_matrix1 = intrinsics[0] @ poses[0].inverse().T[:-1]
    proj_matrix2 = intrinsics[1] @ poses[1].inverse().T[:-1]
    point1 = points[0]
    point2 = points[1]
    A = np.zeros((4, 4))
    A[0] = point1[0] * proj_matrix1[2] - proj_matrix1[0]
    A[1] = point1[1] * proj_matrix1[2] - proj_matrix1[1]
    A[2] = point2[0] * proj_matrix2[2] - proj_matrix2[0]
    A[0] = point2[1] * proj_matrix2[2] - proj_matrix2[1]
    _, s, vt = np.linalg.svd(A)
    return vt[-1, :-1] / vt[-1, -1], s


# https://github.com/colmap/colmap/issues/384
def triangulate_algebraic(observations, poses, K=None):
    assert len(observations) == len(poses)
    if K is None:
        K = np.eye(3)

    A = np.zeros((4, 4))
    points = np.einsum('...ij,...j->...i', np.linalg.inv(K), geometry.homogenize(observations))
    for i in range(len(points)):
        proj_matrix = poses[i].inverse().T[:-1]
        sq_sum = (points[i] ** 2).sum()
        term = (np.eye(3) - np.outer(points[i], points[i]) / sq_sum) @ proj_matrix
        A += term.T @ term
    _, s, vt = np.linalg.svd(A)
    estimated_point = vt[-1]
    return estimated_point[:-1] / estimated_point[-1], s

def triangulate_midpoint(observations, poses, K):
    view_dirs = np.einsum('...li, ...ij, ...j -> ...l', poses.R, np.linalg.inv(K), geometry.homogenize(observations, normalize=False))
    view_dirs /= np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    M = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(len(poses)):
        P = np.eye(3) - np.outer(view_dirs[i], view_dirs[i])
        M += P.T @ P
        b += P.T @ P @ poses.t[i]

    x_midpoint = np.linalg.inv(M) @ b
    return x_midpoint

def add_outliers(observations: np.array, n: int, height, width, c=None, outliers_last=False, check_reprojection=True):
    n_poses = len(observations)
    observations_out = observations.copy()
    if c is None:
        c = np.ones(len(observations))

    if outliers_last:
        outlier_inds = [n_poses - i - 1 for i in range(n)]
    else:
        outlier_inds = np.random.choice(range(n_poses), n, replace=False)

    for i in outlier_inds:
        if check_reprojection:
            diff = 0
            while diff < c[i]:
                random_observation = np.array([np.random.rand() * height, np.random.rand() * width])
                diff = ((random_observation - observations[i]) ** 2).sum()
        else:
            random_observation = np.array([np.random.rand() * height, np.random.rand() * width])
        observations_out[i] = random_observation
    return observations_out, np.array([i not in outlier_inds for i in range(n_poses)])


def get_epipolar_sdr(poses: Se3, observations: np.array, K_inv: np.array):
    n_poses = len(poses)
    x = observations.ravel()
    M = np.block([
        [np.eye(len(x)), -x[:, None]],
        [-x[None], x @ x]
    ])

    fundamental_matrices = geometry.get_fundamental(se3.get_relative(poses), K_inv)
    A = []
    for i in range(n_poses):
        for j in range(i + 1, n_poses):
            A.append(get_epipolar_constraint(fundamental_matrices[i, j], n_poses, i, j))
    A = np.array(A)
    return SDR(M, A, one_hot_matrix(2 * n_poses + 1, -1, -1))


def get_l2_matrix(x, one_first=False):
    """returns matrix M such that [u; 1]^T M [u; 1] = ||u - x||^2"""
    if one_first:
        return np.block([
            [x @ x, -x[None]],
            [-x[:, None], np.eye(len(x))]
        ])
    else:
        return np.block([
            [np.eye(len(x)), -x[:, None]],
            [-x[None], x @ x]
        ])


def get_robust_l2_matrix(x, c=None):
    """returns matrix M such that
    [u.ravel(); theta; 1]^T M [u.ravel(); theta; 1] = sum_i (||u_i - theta_i*x_i|| + c_i*(1 - theta_i))
    """
    n, d = x.shape
    if c is None:
        c = np.ones(n)

    M = np.zeros((n * (d + 1) + 1, n * (d + 1) + 1))
    M[:-1, :-1] = scipy.linalg.block_diag(*[
        np.block([
            [np.eye(d), -x[i][:, None]],
            [-x[i], (x[i] ** 2).sum()]
        ])
        for i in range(n)
    ])
    M[:-1, -1] = -np.kron(c, one_hot(d + 1, d)) * .5
    M[-1, :-1] = -np.kron(c, one_hot(d + 1, d)) * .5
    M[-1, -1] = c.sum()
    return M


def scale_intrinsics(K, scale):
    K_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ])
    return np.einsum('ij, ...jk->...ik', K_scale, K)
