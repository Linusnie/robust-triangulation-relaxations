import numpy as np
from numba import jit, types
from numba.extending import overload


class Quaternion:
    def __init__(self, q):
        self.q = q

    def __getitem__(self, key):
        return Quaternion(self.q[key])

    @property
    def shape(self):
        return self.q.shape[:-1]

    @property
    def w(self):
        return self.q[..., 0]

    @property
    def x(self):
        return self.q[..., 1]

    @property
    def y(self):
        return self.q[..., 2]

    @property
    def z(self):
        return self.q[..., 3]

    @property
    def vec(self):
        return self.q[..., 1:]

    @property
    def left(self):
        left_matrix = np.stack([
            [self.w, -self.x, -self.y, -self.z],
            [self.x, self.w, -self.z, self.y],
            [self.y, self.z, self.w, -self.x],
            [self.z, -self.y, self.x, self.w]
        ])
        if len(self.shape) > 0:
            left_matrix = np.moveaxis(left_matrix, (0, 1), (-2, -1))
        return left_matrix

    @property
    def right(self):
        right_matrix = np.stack([
            [self.w, -self.x, -self.y, -self.z],
            [self.x, self.w, self.z, -self.y],
            [self.y, -self.z, self.w, self.x],
            [self.z, self.y, -self.x, self.w]
        ])
        if len(self.shape) > 0:
            right_matrix = np.moveaxis(right_matrix, (0, 1), (-2, -1))
        return right_matrix

    @property
    def norm(self):
        return np.linalg.norm(self.q, axis=-1)

    @property
    def R(self):
        return q_to_r(self.q)

    @property
    def n(self):
        return r_to_n(self.R)

    @property
    def rotvec(self):
        return r_to_rotvec(self.R)

    def reshape(self, *shape):
        return Quaternion(self.q.reshape(*shape, 4))

    def __mul__(self, other):
        return Quaternion(np.einsum('...ij, ...j -> ...i', self.left, other.q))

    def __pow__(self, power):
        return Quaternion(quaternion_power(self.q, power))

    def __add__(self, other):
        return Quaternion(self.q + other.q)

    def dot(self, other):
        return np.einsum('...i, ...i -> ...', self.q, other.q)

    def interpolate(self, other, alpha):
        if not self.shape == other.shape:
            raise ValueError(f"Shapes must match for interpolation. Got {self.shape}, {other.shape}")

        reverse = np.linalg.norm(self.q + other.q, axis=-1) < np.linalg.norm(self.q - other.q, axis=-1)
        other_signed = Quaternion(other.q.copy())
        other_signed.q[reverse] *= -1
        return self * (self.inverse() * other_signed) ** alpha

    def inverse(self):
        return Quaternion(np.block([self.w[..., None], -self.vec]))

    @staticmethod
    def from_r(R):
        return Quaternion(r_to_q(R))

    def __str__(self):
        return f"quaternion: {self.q}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.shape[0]


# @jit(nopython=True)
def uniform_random_so3(shape):
    """Generate uniform random rotations

    See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf
    """
    rotations_out = np.zeros((*shape, 3, 3))
    for idx in np.ndindex(shape):
        x1, x2 = np.random.random(2) * 2 * np.pi
        x3 = np.random.random()
        rz = np.array([
            [np.cos(x1), np.sin(x1), 0.],
            [-np.sin(x1), np.cos(x1), 0.],
            [0., 0., 1.]
        ])
        v = np.array([
            np.cos(x2) * np.sqrt(x3),
            np.sin(x2) * np.sqrt(x3),
            np.sqrt(1 - x3)
        ])
        rotations_out[idx] = - (np.eye(3) - 2 * np.outer(v, v)) @ rz
    return rotations_out


def axis_rotation(angle, axis):
    sin, cos = np.sin(angle), np.cos(angle)
    r_out = np.eye(3)
    i, j = {"x": [1, 2], "y": [0, 2], "z": [0, 1]}[axis]
    r_out[i, i], r_out[i, j] = cos, -sin
    r_out[j, i], r_out[j, j] = sin, cos
    return r_out


def interpolate_quaternions(q, timestamps, target_timestamps):
    """Spherical interpolation of quaternions.

    Args:
        q: (Nx4 array) quaternions to interpolate from
        timestamps (sorted length N array) timestamps corresponding to each quaternion
        target_timestamps (sorted length M array) timestamps to interpolate to

    """
    if (target_timestamps[0] < timestamps[0]) or (target_timestamps[-1] > timestamps[-1]):
        raise ValueError(f"target range ({target_timestamps[0]}, {target_timestamps[-1]}) is not contained in "
                         f"timestamp range ({timestamps[0], timestamps[-1]})")
    target_timestamps = np.array(target_timestamps)
    q_out = np.zeros((len(target_timestamps), 4))
    start = 0
    for i in range(len(timestamps)-1):
        # look for target timestamps between timestamps[i] and timestamps[i+1] and interpolate
        stop = np.searchsorted(target_timestamps, timestamps[i+1])
        if start != stop:
            dt = timestamps[i+1] - timestamps[i]
            q_out[start:stop] = q[i].interpolate(
                q[i+1], (target_timestamps[start:stop] - timestamps[i]) / dt).q
        start = stop
    return Quaternion(q_out)


# @jit(nopython=True)
def _quaternion_power(q, power):
    q_out = np.zeros(4)
    half_angle = np.arccos(min(1, max(-1, q[0])))
    q_out[0] = np.cos(half_angle * power)
    imag_norm = np.linalg.norm(q[1:])
    if imag_norm != 0:
        q_out[1:] = q[1:] * np.sin(half_angle * power) / imag_norm
    return q_out

# allow calling np.array on arrays for type-promotion
# ser https://github.com/numba/numba/issues/4470
@overload(np.array)
def np_array_overload(x):
    if isinstance(x, types.Array):
        def impl(x):
            return np.copy(x)
        return impl


# @jit(nopython=True)
def quaternion_power(q_array, powers):
    powers = np.array(powers)
    q_out = np.zeros((*powers.shape, *q_array.shape[:-1], 4))
    for power_idx in np.ndindex(powers.shape):
        for q_idx in np.ndindex(q_array.shape[:-1]):
            q_out[power_idx][q_idx] = _quaternion_power(q_array[q_idx], powers[power_idx])
    return q_out


def copysign(a, b):
    a[np.sign(a) != np.sign(b)] *= -1


def r_to_q(R):
    q_out = np.zeros((*R.shape[:-2], 4))
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    q_out[..., 0] = np.sqrt(np.maximum(1 + R00 + R11 + R22, 0)) / 2
    q_out[..., 1] = np.sqrt(np.maximum(1 + R00 - R11 - R22, 0)) / 2
    q_out[..., 2] = np.sqrt(np.maximum(1 - R00 + R11 - R22, 0)) / 2
    q_out[..., 3] = np.sqrt(np.maximum(1 - R00 - R11 + R22, 0)) / 2
    copysign(q_out[..., 1], R21 - R12)
    copysign(q_out[..., 2], R02 - R20)
    copysign(q_out[..., 3], R10 - R01)
    return q_out


# @jit(nopython=True)
def n_to_q(n):
    q_out = np.zeros((*n.shape[:-1], 4))
    for idx in np.ndindex(*n.shape[:-1]):
        theta = np.linalg.norm(n[idx])
        if theta != 0:
            q_out[idx][0] = np.cos(theta / 2)
            q_out[idx][1:] = np.sin(theta / 2) * n[idx] / theta
        else:
            q_out[idx][0] = 1
    return q_out


# @jit(nopython=True)
def r_to_n(R):
    n = np.zeros((*R.shape[:-2], 4))
    n[..., 0] = np.arccos((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2) * 180 / np.pi
    n[..., 1] = R[..., 2, 1] - R[..., 1, 2]
    n[..., 2] = R[..., 0, 2] - R[..., 2, 0]
    n[..., 3] = R[..., 1, 0] - R[..., 0, 1]
    n[..., 1:] /= np.sqrt((n[..., 1:] ** 2).sum(axis=-1)).reshape(-1, 1)
    return n


def rotvec_to_r(n):
    cos = np.cos(np.linalg.norm(n, axis=-1))
    sin = np.sin(np.linalg.norm(n, axis=-1))
    nhat = n / np.linalg.norm(n, axis=-1)[..., None]

    R = np.zeros((*n.shape[:-1], 3, 3))
    R += cos[..., None, None] * np.tile(np.eye(3), (*n.shape[:-1], 1, 1))
    R += sin[..., None, None] * skew(nhat)
    R += (1 - cos[..., None, None]) * np.einsum('...i,...j->...ij', nhat, nhat)
    R[sin == 0] = np.eye(3)
    return R


def r_to_rotvec(R):
    n = np.zeros((*R.shape[:-2], 3))
    angle = np.arccos(np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2, -1, 1))
    n[..., 0] = R[..., 2, 1] - R[..., 1, 2]
    n[..., 1] = R[..., 0, 2] - R[..., 2, 0]
    n[..., 2] = R[..., 1, 0] - R[..., 0, 1]
    norms = np.linalg.norm(n, axis=-1)
    nonzero_indices = norms != 0
    n[nonzero_indices] *= (angle[nonzero_indices] / norms[nonzero_indices])[..., None]
    return n


# @jit(nopython=True)
def q_to_r(q, real_first=True):
    if real_first:
        a, b, c, d = [q[..., i] for i in range(4)]
    else:
        b, c, d, a = [q[..., i] for i in range(4)]

    R_out = np.zeros((*q.shape[:-1], 3, 3))
    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    ab = 2 * a * b
    ac = 2 * a * c
    ad = 2 * a * d
    bc = 2 * b * c
    bd = 2 * b * d
    cd = 2 * c * d

    R_out[..., 0, 0] = a2 + b2 - c2 - d2
    R_out[..., 0, 1] = bc + ad
    R_out[..., 0, 2] = bd - ac
    R_out[..., 1, 0] = bc - ad
    R_out[..., 1, 1] = a2 - b2 + c2 - d2
    R_out[..., 1, 2] = cd + ab
    R_out[..., 2, 0] = bd + ac
    R_out[..., 2, 1] = cd - ab
    R_out[..., 2, 2] = a2 - b2 - c2 + d2
    return np.swapaxes(R_out, -1, -2)


# @jit(nopython=True)
def skew(n):
    skew_matrix = np.zeros((*n.shape[:-1], 3, 3))
    skew_matrix[..., 2, 1] = n[..., 0]
    skew_matrix[..., 1, 2] = -n[..., 0]
    skew_matrix[..., 0, 2] = n[..., 1]
    skew_matrix[..., 2, 0] = -n[..., 1]
    skew_matrix[..., 1, 0] = n[..., 2]
    skew_matrix[..., 0, 1] = -n[..., 2]
    return skew_matrix


def get_minimal_rotation(a, b):
    """Get the rotation matrix with minimal angle which satisfies Ra = b"""
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    sin = np.linalg.norm(v)
    cos = np.dot(a, b)
    vx = skew(v)
    return np.eye(3) + vx + vx @ vx * (1 - cos) / sin**2


def get_rotaitons_facing_point(point, origins):
    n_poses = len(origins)
    d = (point - origins) / np.linalg.norm(point - origins, axis=-1)[:, None]
    R = np.zeros((n_poses, 3, 3))
    R[..., -1] = d
    R[..., 0] = np.cross(d, np.random.randn(n_poses, 3))
    R[..., 0] /= np.linalg.norm(R[..., 0], axis=-1)[..., None]
    R[..., 1] = np.cross(R[..., 2], R[..., 0])
    return R


def rot_x(angle):
    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([
        [1, 0, 0],
        [0, cos, -sin],
        [0, sin, cos]
    ])


def rot_y(angle):
    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([
        [cos, 0, -sin],
        [0, 1, 0],
        [sin, 0, cos]
    ])


def rot_z(angle):
    sin, cos = np.sin(angle), np.cos(angle)
    return np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ])
