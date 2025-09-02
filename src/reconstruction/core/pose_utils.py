from __future__ import annotations

import numpy as np
from numpy.linalg import norm

_EPS = 1e-12
_PI = np.pi


def is_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if R is a valid rotation matrix (orthonormal with det ≈ +1)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    RtR = R.T @ R
    I = np.eye(3)
    if norm(RtR - I) > tol:
        return False
    if abs(np.linalg.det(R) - 1.0) > tol:
        return False
    return True


def _as_unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = norm(v)
    if n < _EPS:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion (supports shapes (...,4))."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must have last dimension 4.")
    n = norm(q, axis=-1, keepdims=True)
    if np.any(n < _EPS):
        raise ValueError("Zero-norm quaternion.")
    return q / n


def _stack_quat(w: float, x: float, y: float, z: float, order: str) -> np.ndarray:
    if order == "wxyz":
        return np.array([w, x, y, z], dtype=float)
    elif order == "xyzw":
        return np.array([x, y, z, w], dtype=float)
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")


def _split_quat(q: np.ndarray, order: str):
    q = np.asarray(q, dtype=float).reshape(4)
    if order not in ("wxyz", "xyzw"):
        raise ValueError("order must be 'wxyz' or 'xyzw'")
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    return w, x, y, z


# ----------- Q <-> Rotation Matrix --------------------

def q_to_Rotation(q: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Quaternion -> 3x3 rotation matrix.

    Args:
        q: quaternion [w,x,y,z] if order='wxyz' or [x,y,z,w] if order='xyzw'.
        order: 'wxyz' (default) or 'xyzw'.
    """
    w, x, y, z = _split_quat(quat_normalize(q), order)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)
    return R


def Rotation_to_q(R: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """3x3 rotation matrix -> quaternion (scalar-first by default)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    if not is_rotation_matrix(R):
        # best-effort orthonormalization
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if not is_rotation_matrix(R):
            raise ValueError("Input is not a valid rotation matrix even after SVD orthonormalization.")

    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2.0  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = _stack_quat(qw, qx, qy, qz, order)
    return quat_normalize(q)




# ----------------- Euler <-> Rotation Matrix --------------------------

def euler_to_Rotation(angles, order: str = "xyz", degrees: bool = False) -> np.ndarray:
    """Euler angles -> rotation matrix (intrinsic).

    order='xyz' means roll->pitch->yaw (Rx @ Ry @ Rz). 'zyx' supported too.
    """
    if len(order) != 3 or any(a not in "xyz" for a in order):
        raise ValueError("order must be a 3-char string from 'x','y','z', e.g., 'xyz','zyx'")
    a = np.asarray(angles, dtype=float).reshape(3)
    if degrees:
        a = np.deg2rad(a)

    s = np.sin(a)
    c = np.cos(a)

    def Rx(ca, sa):
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], float)

    def Ry(ca, sa):
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], float)

    def Rz(ca, sa):
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], float)

    R_ops = {"x": Rx, "y": Ry, "z": Rz}
    R = np.eye(3)
    for idx, axis in enumerate(order):
        R = R @ R_ops[axis](c[idx], s[idx])
    return R


def Rotation_to_euler(R: np.ndarray, order: str = "xyz", degrees: bool = False) -> np.ndarray:
    """Rotation matrix -> Euler angles (intrinsic). Supports 'xyz' and 'zyx'."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3,3):
        raise ValueError("R must be 3x3")
    if not is_rotation_matrix(R):
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

    if order == "xyz":
        sy = -R[2,0]
        sy = np.clip(sy, -1.0, 1.0)
        pitch = np.arcsin(sy)
        if abs(sy) < 1 - 1e-7:
            roll  = np.arctan2(R[2,1], R[2,2])
            yaw   = np.arctan2(R[1,0], R[0,0])
        else:
            roll  = 0.0
            yaw   = np.arctan2(-R[0,1], R[1,1])
        angles = np.array([roll, pitch, yaw])
    elif order == "zyx":
        sy = -R[0,2]
        sy = np.clip(sy, -1.0, 1.0)
        pitch = np.arcsin(sy)
        if abs(sy) < 1 - 1e-7:
            yaw   = np.arctan2(R[0,1], R[0,0])
            roll  = np.arctan2(R[1,2], R[2,2])
        else:
            yaw   = 0.0
            roll  = np.arctan2(-R[2,1], R[1,1])
        angles = np.array([roll, pitch, yaw])
    else:
        raise NotImplementedError("Rotation_to_euler supports only 'xyz' and 'zyx'")

    if degrees:
        angles = np.rad2deg(angles)
    return angles


def q_to_euler(q: np.ndarray, order: str = "xyz", degrees: bool = False, q_order: str = "wxyz") -> np.ndarray:
    """Quaternion -> Euler (via rotation matrix)."""
    R = q_to_Rotation(q, order=q_order)
    return Rotation_to_euler(R, order=order, degrees=degrees)


def euler_to_q(angles, order: str = "xyz", degrees: bool = False, q_order: str = "wxyz") -> np.ndarray:
    """Euler -> Quaternion (via rotation matrix)."""
    R = euler_to_Rotation(angles, order=order, degrees=degrees)
    return Rotation_to_q(R, order=q_order)
