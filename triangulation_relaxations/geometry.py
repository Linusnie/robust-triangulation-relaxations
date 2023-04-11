from triangulation_relaxations import so3, se3
import numpy as np


def reproject(point: np.array, poses: se3.Se3, K: np.array):
    x = transform(poses.inverse(), point, K)
    x = x[..., :2] / x[..., 2:]
    return x


def batch_matvec(A, b, c=None):
    # A(dims1..., i, j), b(dims2..., j), c(dims1..., j) -> Ab+c(dims1..., dims2..., i)
    product = np.einsum('...ij, ...kj->...ki', A, b.reshape(-1, b.shape[-1]))
    if c is not None:
        product += c[..., None, :]
    return product.reshape((*A.shape[:-2], *b.shape[:-1], -1))


def transform(poses: se3.Se3, points: np.array, K: np.array):
    KR = np.einsum('...ij, ...jk->...ik', K, poses.R)
    Kt = np.einsum('...ij, ...j->...i', K, poses.t)
    points_transformed = batch_matvec(KR, points, Kt)
    return points_transformed


def homogenize(x: np.array, normalize: bool = False):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    x = x.reshape((*shape[:-1], shape[-1] + 1))
    if normalize:
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x


def get_essential(poses: se3.Se3):
    return np.einsum('...ij,...jk->...ik', so3.skew(poses.t), poses.R)


def get_relative_fundamental(poses: se3.Se3, K_inv: np.array):
    n_poses = len(poses)
    rel_poses = se3.get_relative(poses)
    F = np.zeros((n_poses, n_poses, 3, 3))
    for i in range(n_poses):
        for j in range(n_poses):
            if len(K_inv.shape) == 3:
                F[i, j] = K_inv[i].T @ get_essential(rel_poses[i, j]) @ K_inv[j]
            else:
                F[i, j] = K_inv.T @ get_essential(rel_poses[i, j]) @ K_inv
    return F


def get_fundamental(poses: se3.Se3, K_inv: np.array):
    return np.einsum('...ij,...jk,...kl->...il', K_inv.T, get_essential(poses), K_inv)
