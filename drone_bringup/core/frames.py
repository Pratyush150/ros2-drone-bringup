"""Rotation math and the exact PX4 <-> ROS frame transform chain.

Scope
-----
Quaternion/Euler conversions, body-frame FRD<->FLU conversion, and the two
composed transforms that take a PX4 attitude (NED world, FRD body) to a ROS
attitude (ENU world, FLU body) and back.

Conventions used throughout (state them, because half of all frame bugs are two
libraries silently disagreeing about one of these):

* Quaternions are ``(w, x, y, z)`` -- **scalar first**. ROS messages
  (``geometry_msgs/Quaternion``) are scalar *last* ``(x, y, z, w)``; PX4 uORB
  ``vehicle_attitude.q`` is scalar *first*. Use :func:`quat_to_ros_xyzw` and
  :func:`quat_from_ros_xyzw` at the message boundary and nowhere else.
* Euler angles are intrinsic Tait-Bryan Z-Y-X: yaw about Z, then pitch about
  the new Y, then roll about the new X. This matches REP-103, PX4, and every
  autopilot ground station. It does *not* match a naive "rotate about fixed
  axes" reading.
* Rotations act on column vectors: ``v_world = R * v_body``. A quaternion
  ``q_wb`` therefore rotates a vector expressed in the body frame into the
  world frame.

The two world/body conventions
------------------------------
=============  =====================  ==========================
Frame          World                  Body
=============  =====================  ==========================
ROS / REP-103  ENU (x East, y North,  FLU (x Forward, y Left,
               z Up)                  z Up)
PX4 / MAVLink  NED (x North, y East,  FRD (x Forward, y Right,
               z Down)                z Down)
=============  =====================  ==========================

Converting an *attitude* between these needs two rotations, not one:

    q_enu_flu = q_ned_to_enu * q_ned_frd * q_frd_to_flu

where ``q_ned_to_enu`` is a fixed 180 deg rotation about the axis
``(1/sqrt(2), 1/sqrt(2), 0)`` -- the same "swap x/y, negate z" involution as in
:mod:`drone_bringup.core.geodesy` -- and ``q_frd_to_flu`` is a 180 deg rotation
about body X (roll by pi). Both are involutions, so the same composition runs
in reverse. Getting only one of the two applied is the classic failure: level
hover looks fine, but pitch and yaw come out mirrored the moment you bank.

Pure Python, no ROS, no numpy.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

__all__ = [
    "Quaternion",
    "quat_identity",
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_norm",
    "quat_from_euler",
    "quat_to_euler",
    "quat_rotate_vector",
    "quat_to_matrix",
    "quat_to_ros_xyzw",
    "quat_from_ros_xyzw",
    "frd_to_flu",
    "flu_to_frd",
    "Q_NED_TO_ENU",
    "Q_FRD_TO_FLU",
    "px4_attitude_to_ros",
    "ros_attitude_to_px4",
    "angular_distance",
]

Quaternion = Tuple[float, float, float, float]
"""A scalar-first quaternion ``(w, x, y, z)``."""

_EPS = 1e-12


# --- basic quaternion algebra -----------------------------------------------
def quat_identity() -> Quaternion:
    """Return the identity rotation ``(1, 0, 0, 0)``."""
    return (1.0, 0.0, 0.0, 0.0)


def quat_norm(q: Sequence[float]) -> float:
    """Euclidean norm of a quaternion."""
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def quat_normalize(q: Sequence[float]) -> Quaternion:
    """Return ``q`` scaled to unit norm.

    Raises:
        ValueError: If ``q`` has (near) zero norm and cannot be normalised.
    """
    n = quat_norm(q)
    if n < _EPS:
        raise ValueError("cannot normalize a zero-norm quaternion")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_conjugate(q: Sequence[float]) -> Quaternion:
    """Conjugate ``(w, -x, -y, -z)``. For a unit quaternion this is the inverse."""
    return (q[0], -q[1], -q[2], -q[3])


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    """Hamilton product ``a * b``.

    Composition order is "apply ``b`` first, then ``a``", consistent with
    ``R(a*b) = R(a) @ R(b)``.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# --- Euler <-> quaternion ----------------------------------------------------
def quat_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Build a quaternion from intrinsic Z-Y-X Tait-Bryan angles (radians)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_to_euler(q: Sequence[float]) -> Tuple[float, float, float]:
    """Recover intrinsic Z-Y-X ``(roll, pitch, yaw)`` in radians from a quaternion.

    Gimbal lock (pitch at +/-90 deg) is handled by clamping the pitch argument
    and folding the degenerate roll/yaw pair into yaw, so the result stays
    finite instead of returning NaN. That matters because a fixed-wing in a
    vertical climb, or a gimbal pointed straight down, will hit it.
    """
    w, x, y, z = quat_normalize(q)
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0 - 1e-9:
        pitch = math.copysign(math.pi / 2.0, sinp)
        # Degenerate: only (roll +/- yaw) is observable. Report roll = 0.
        roll = 0.0
        yaw = math.copysign(2.0, sinp) * math.atan2(x, w)
        return (roll, pitch, _wrap_pi(yaw))
    pitch = math.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def _wrap_pi(angle: float) -> float:
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


# --- rotation application ----------------------------------------------------
def quat_rotate_vector(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate a 3-vector by a quaternion: ``v' = q * v * q^-1``."""
    qn = quat_normalize(q)
    vq = (0.0, float(v[0]), float(v[1]), float(v[2]))
    out = quat_multiply(quat_multiply(qn, vq), quat_conjugate(qn))
    return (out[1], out[2], out[3])


def quat_to_matrix(q: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
    """Return the 3x3 rotation matrix as a tuple of row tuples."""
    w, x, y, z = quat_normalize(q)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Smallest rotation angle in radians between two orientations.

    Accounts for double cover: ``q`` and ``-q`` are the same rotation, so the
    result is always in ``[0, pi]``.
    """
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


# --- message-boundary ordering ----------------------------------------------
def quat_to_ros_xyzw(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """Reorder scalar-first ``(w, x, y, z)`` to ROS scalar-last ``(x, y, z, w)``."""
    return (q[1], q[2], q[3], q[0])


def quat_from_ros_xyzw(q: Sequence[float]) -> Quaternion:
    """Reorder ROS scalar-last ``(x, y, z, w)`` to scalar-first ``(w, x, y, z)``."""
    return (q[3], q[0], q[1], q[2])


# --- body-frame conversion ---------------------------------------------------
def frd_to_flu(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert a body-frame vector from FRD (PX4) to FLU (ROS).

    ``(fwd, left, up) = (fwd, -right, -down)`` -- a 180 deg roll about body X.
    Self-inverse.
    """
    return (x, -y, -z)


def flu_to_frd(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert a body-frame vector from FLU (ROS) to FRD (PX4)."""
    return (x, -y, -z)


# --- the fixed transform chain ----------------------------------------------
_SQRT1_2 = math.sqrt(0.5)

Q_NED_TO_ENU: Quaternion = (0.0, _SQRT1_2, _SQRT1_2, 0.0)
"""World-frame rotation NED -> ENU: 180 deg about ``(1,1,0)/sqrt(2)``.

Maps ``x_ned(North) -> y_enu(North)``, ``y_ned(East) -> x_enu(East)``,
``z_ned(Down) -> -z_enu(Up)``. Involution, so it is also ENU -> NED.
"""

Q_FRD_TO_FLU: Quaternion = (0.0, 1.0, 0.0, 0.0)
"""Body-frame rotation FRD -> FLU: 180 deg about body X. Involution."""


def px4_attitude_to_ros(q_ned_frd: Sequence[float]) -> Quaternion:
    """Convert a PX4 attitude quaternion to the ROS convention.

    Args:
        q_ned_frd: Scalar-first quaternion rotating FRD body vectors into the
            NED world frame -- i.e. exactly what PX4 puts in
            ``vehicle_attitude.q`` / MAVLink ``ATTITUDE_QUATERNION``.

    Returns:
        Scalar-first quaternion rotating FLU body vectors into the ENU world
        frame, ready for ``geometry_msgs/Quaternion`` after
        :func:`quat_to_ros_xyzw`.

    The composition is ``Q_NED_TO_ENU * q * Q_FRD_TO_FLU``: pre-multiply to
    re-express the *world* frame, post-multiply to re-express the *body* frame.
    Doing only one of the two is the single most common frame bug in
    PX4<->ROS 2 bridges.
    """
    return quat_normalize(
        quat_multiply(quat_multiply(Q_NED_TO_ENU, q_ned_frd), Q_FRD_TO_FLU)
    )


def ros_attitude_to_px4(q_enu_flu: Sequence[float]) -> Quaternion:
    """Inverse of :func:`px4_attitude_to_ros`.

    Both fixed rotations are involutions, so the composition is identical --
    but keep two named functions so the direction is obvious at the call site.
    """
    return quat_normalize(
        quat_multiply(quat_multiply(Q_NED_TO_ENU, q_enu_flu), Q_FRD_TO_FLU)
    )
