"""Tests for quaternion math and the PX4 <-> ROS transform chain.

The interesting assertions are the ones that check *physical meaning*: a PX4
attitude with the nose pitched up must come out as a ROS attitude with the body
x-axis pointing up in ENU, whatever the sign conventions do along the way.
"""

import math

import pytest

from drone_bringup.core.frames import (
    Q_FRD_TO_FLU,
    Q_NED_TO_ENU,
    angular_distance,
    flu_to_frd,
    frd_to_flu,
    px4_attitude_to_ros,
    quat_conjugate,
    quat_from_euler,
    quat_from_ros_xyzw,
    quat_identity,
    quat_multiply,
    quat_norm,
    quat_normalize,
    quat_rotate_vector,
    quat_to_euler,
    quat_to_matrix,
    quat_to_ros_xyzw,
    ros_attitude_to_px4,
)

DEG = math.pi / 180.0


class TestQuaternionAlgebra:
    """Basic algebra: identity, product, conjugate, normalisation."""

    def test_identity_is_unit(self):
        assert quat_norm(quat_identity()) == pytest.approx(1.0)

    def test_identity_is_a_multiplicative_identity(self):
        q = quat_from_euler(0.3, -0.2, 1.1)
        assert quat_multiply(q, quat_identity()) == pytest.approx(q)
        assert quat_multiply(quat_identity(), q) == pytest.approx(q)

    def test_conjugate_undoes_rotation(self):
        q = quat_from_euler(0.4, 0.2, -0.9)
        product = quat_multiply(q, quat_conjugate(q))
        assert product == pytest.approx(quat_identity(), abs=1e-12)

    def test_normalize_scales_to_unit(self):
        q = quat_normalize((2.0, 0.0, 0.0, 0.0))
        assert q == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_normalize_rejects_zero(self):
        with pytest.raises(ValueError, match="zero-norm"):
            quat_normalize((0.0, 0.0, 0.0, 0.0))

    def test_product_is_not_commutative(self):
        a = quat_from_euler(0.0, 0.0, math.pi / 2)
        b = quat_from_euler(math.pi / 2, 0.0, 0.0)
        assert quat_multiply(a, b) != pytest.approx(quat_multiply(b, a))

    def test_composition_matches_sequential_rotation(self):
        a = quat_from_euler(0.0, 0.0, math.pi / 2)  # yaw 90
        b = quat_from_euler(0.0, math.pi / 2, 0.0)  # pitch 90
        v = (1.0, 0.0, 0.0)
        composed = quat_rotate_vector(quat_multiply(a, b), v)
        stepwise = quat_rotate_vector(a, quat_rotate_vector(b, v))
        assert composed == pytest.approx(stepwise, abs=1e-12)


class TestEulerConversion:
    """Intrinsic Z-Y-X Tait-Bryan angles."""

    @pytest.mark.parametrize(
        "rpy",
        [
            (0.0, 0.0, 0.0),
            (10 * DEG, 0.0, 0.0),
            (0.0, -25 * DEG, 0.0),
            (0.0, 0.0, 179 * DEG),
            (15 * DEG, -20 * DEG, 130 * DEG),
            (-45 * DEG, 30 * DEG, -170 * DEG),
        ],
    )
    def test_euler_round_trip(self, rpy):
        q = quat_from_euler(*rpy)
        assert quat_to_euler(q) == pytest.approx(rpy, abs=1e-9)

    def test_yaw_only_rotates_x_towards_y(self):
        q = quat_from_euler(0.0, 0.0, 90 * DEG)
        assert quat_rotate_vector(q, (1.0, 0.0, 0.0)) == pytest.approx(
            (0.0, 1.0, 0.0), abs=1e-12
        )

    def test_pitch_sign_convention(self):
        # In a right-handed frame with x forward and z up, a positive pitch
        # (rotation about +y) tips the nose DOWN. That is FLU. In FRD, where z
        # is down, the same sign tips the nose up. This is the sign flip that
        # makes PX4 pitch and ROS pitch disagree.
        q = quat_from_euler(0.0, 20 * DEG, 0.0)
        x, y, z = quat_rotate_vector(q, (1.0, 0.0, 0.0))
        assert z < 0.0
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_gimbal_lock_returns_finite_angles(self):
        q = quat_from_euler(0.0, math.pi / 2, 0.0)
        roll, pitch, yaw = quat_to_euler(q)
        assert math.isfinite(roll) and math.isfinite(pitch) and math.isfinite(yaw)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-6)

    def test_double_cover_gives_the_same_euler_angles(self):
        q = quat_from_euler(0.2, -0.3, 0.9)
        negated = tuple(-c for c in q)
        assert quat_to_euler(negated) == pytest.approx(quat_to_euler(q), abs=1e-12)


class TestRotationMatrix:
    """Matrix form, used mostly for sanity-checking the quaternion path."""

    def test_identity_matrix(self):
        rows = quat_to_matrix(quat_identity())
        assert rows[0] == pytest.approx((1.0, 0.0, 0.0))
        assert rows[1] == pytest.approx((0.0, 1.0, 0.0))
        assert rows[2] == pytest.approx((0.0, 0.0, 1.0))

    def test_matrix_is_orthonormal(self):
        rows = quat_to_matrix(quat_from_euler(0.3, -0.7, 1.4))
        for row in rows:
            assert math.sqrt(sum(c * c for c in row)) == pytest.approx(1.0)
        # Rows are mutually orthogonal.
        assert sum(a * b for a, b in zip(rows[0], rows[1])) == pytest.approx(0.0, abs=1e-12)
        assert sum(a * b for a, b in zip(rows[1], rows[2])) == pytest.approx(0.0, abs=1e-12)

    def test_matrix_agrees_with_quaternion_rotation(self):
        q = quat_from_euler(0.1, 0.5, -1.2)
        rows = quat_to_matrix(q)
        v = (2.0, -1.0, 0.5)
        by_matrix = tuple(sum(r[i] * v[i] for i in range(3)) for r in rows)
        assert quat_rotate_vector(q, v) == pytest.approx(by_matrix, abs=1e-12)


class TestMessageOrdering:
    """Scalar-first (PX4) vs scalar-last (ROS) component order."""

    def test_to_ros_moves_w_to_the_end(self):
        assert quat_to_ros_xyzw((1.0, 2.0, 3.0, 4.0)) == (2.0, 3.0, 4.0, 1.0)

    def test_from_ros_moves_w_to_the_front(self):
        assert quat_from_ros_xyzw((2.0, 3.0, 4.0, 1.0)) == (1.0, 2.0, 3.0, 4.0)

    def test_ordering_round_trip(self):
        q = quat_from_euler(0.2, 0.3, 0.4)
        assert quat_from_ros_xyzw(quat_to_ros_xyzw(q)) == pytest.approx(q)


class TestBodyFrame:
    """FRD (PX4 body) <-> FLU (ROS body)."""

    def test_forward_is_unchanged(self):
        assert frd_to_flu(1.0, 0.0, 0.0) == (1.0, 0.0, 0.0)

    def test_right_becomes_negative_left(self):
        assert frd_to_flu(0.0, 1.0, 0.0) == (0.0, -1.0, 0.0)

    def test_down_becomes_negative_up(self):
        assert frd_to_flu(0.0, 0.0, 1.0) == (0.0, 0.0, -1.0)

    def test_body_conversion_is_an_involution(self):
        v = (1.5, -2.5, 3.5)
        assert flu_to_frd(*frd_to_flu(*v)) == v


class TestFixedRotations:
    """The two constant rotations that make up the PX4 <-> ROS chain."""

    def test_ned_to_enu_is_unit(self):
        assert quat_norm(Q_NED_TO_ENU) == pytest.approx(1.0)

    def test_ned_to_enu_swaps_north_and_east(self):
        # NED x (North) must land on ENU y (North).
        assert quat_rotate_vector(Q_NED_TO_ENU, (1.0, 0.0, 0.0)) == pytest.approx(
            (0.0, 1.0, 0.0), abs=1e-12
        )
        # NED y (East) must land on ENU x (East).
        assert quat_rotate_vector(Q_NED_TO_ENU, (0.0, 1.0, 0.0)) == pytest.approx(
            (1.0, 0.0, 0.0), abs=1e-12
        )

    def test_ned_to_enu_flips_down_to_up(self):
        assert quat_rotate_vector(Q_NED_TO_ENU, (0.0, 0.0, 1.0)) == pytest.approx(
            (0.0, 0.0, -1.0), abs=1e-12
        )

    def test_both_fixed_rotations_are_involutions(self):
        # q*q comes out as -identity, not +identity: a 180 degree rotation
        # squares to the negated quaternion. Same rotation, opposite sign --
        # which is exactly why you compare orientations with an angular metric
        # and never with component equality.
        for q in (Q_NED_TO_ENU, Q_FRD_TO_FLU):
            squared = quat_multiply(q, q)
            assert angular_distance(squared, quat_identity()) == pytest.approx(
                0.0, abs=1e-9
            )
            assert squared[0] == pytest.approx(-1.0, abs=1e-12)

    def test_frd_to_flu_is_a_roll_of_pi(self):
        roll, _, _ = quat_to_euler(Q_FRD_TO_FLU)
        assert abs(roll) == pytest.approx(math.pi, abs=1e-9)


class TestAttitudeChain:
    """The composed PX4 -> ROS attitude transform, checked physically."""

    def test_level_nose_north_becomes_enu_yaw_ninety(self):
        # PX4 identity attitude = FRD aligned with NED = level, nose North.
        # In ENU/FLU that is yaw 90 deg (North is 90 deg CCW from East).
        q_ros = px4_attitude_to_ros(quat_identity())
        roll, pitch, yaw = quat_to_euler(q_ros)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw == pytest.approx(90 * DEG, abs=1e-9)

    def test_ned_heading_thirty_becomes_enu_yaw_sixty(self):
        # yaw_enu = 90 - yaw_ned.
        q_ros = px4_attitude_to_ros(quat_from_euler(0.0, 0.0, 30 * DEG))
        _, _, yaw = quat_to_euler(q_ros)
        assert yaw == pytest.approx(60 * DEG, abs=1e-9)

    def test_ned_heading_ninety_becomes_enu_yaw_zero(self):
        q_ros = px4_attitude_to_ros(quat_from_euler(0.0, 0.0, 90 * DEG))
        _, _, yaw = quat_to_euler(q_ros)
        assert yaw == pytest.approx(0.0, abs=1e-9)

    def test_nose_up_stays_nose_up(self):
        # PX4 pitch +10 deg is nose up. After conversion, the ROS body x-axis
        # expressed in ENU must have a positive Up component.
        q_px4 = quat_from_euler(0.0, 10 * DEG, 0.0)
        _, _, up = quat_rotate_vector(px4_attitude_to_ros(q_px4), (1.0, 0.0, 0.0))
        assert up > 0.0
        assert up == pytest.approx(math.sin(10 * DEG), abs=1e-9)

    def test_pitch_sign_flips_between_conventions(self):
        # The physical attitude is identical; only the Euler sign changes,
        # because FLU's y (Left) points the opposite way to FRD's y (Right).
        q_px4 = quat_from_euler(0.0, 10 * DEG, 0.0)
        _, pitch_ros, _ = quat_to_euler(px4_attitude_to_ros(q_px4))
        assert pitch_ros == pytest.approx(-10 * DEG, abs=1e-9)

    def test_roll_sign_is_preserved(self):
        # Roll is about the forward axis, which both conventions agree on.
        q_px4 = quat_from_euler(12 * DEG, 0.0, 0.0)
        roll_ros, _, _ = quat_to_euler(px4_attitude_to_ros(q_px4))
        assert roll_ros == pytest.approx(12 * DEG, abs=1e-9)

    def test_nose_north_body_x_points_north_in_enu(self):
        east, north, up = quat_rotate_vector(
            px4_attitude_to_ros(quat_identity()), (1.0, 0.0, 0.0)
        )
        assert (east, north, up) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)

    @pytest.mark.parametrize(
        "rpy",
        [
            (0.0, 0.0, 0.0),
            (5 * DEG, -8 * DEG, 40 * DEG),
            (-30 * DEG, 15 * DEG, -120 * DEG),
            (0.0, 0.0, 179 * DEG),
        ],
    )
    def test_chain_round_trip(self, rpy):
        q_px4 = quat_from_euler(*rpy)
        back = ros_attitude_to_px4(px4_attitude_to_ros(q_px4))
        assert angular_distance(back, q_px4) == pytest.approx(0.0, abs=1e-9)

    def test_applying_only_the_world_rotation_is_wrong(self):
        # The failure mode the module docstring warns about: pre-multiplying the
        # NED->ENU rotation but forgetting the FRD->FLU body rotation. It looks
        # right in level hover and is wrong the moment the vehicle rolls.
        q_px4 = quat_from_euler(20 * DEG, 0.0, 0.0)
        half_done = quat_multiply(Q_NED_TO_ENU, q_px4)
        correct = px4_attitude_to_ros(q_px4)
        assert angular_distance(half_done, correct) > 1 * DEG

    def test_half_done_conversion_mirrors_roll(self):
        q_px4 = quat_from_euler(20 * DEG, 0.0, 0.0)
        roll_wrong, _, _ = quat_to_euler(quat_multiply(Q_NED_TO_ENU, q_px4))
        roll_right, _, _ = quat_to_euler(px4_attitude_to_ros(q_px4))
        assert roll_right == pytest.approx(20 * DEG, abs=1e-9)
        assert roll_wrong != pytest.approx(roll_right, abs=1e-3)


class TestAngularDistance:
    """Shortest-angle metric, accounting for double cover."""

    def test_zero_for_identical_orientations(self):
        q = quat_from_euler(0.1, 0.2, 0.3)
        assert angular_distance(q, q) == pytest.approx(0.0, abs=1e-9)

    def test_zero_for_negated_quaternion(self):
        q = quat_from_euler(0.1, 0.2, 0.3)
        assert angular_distance(q, tuple(-c for c in q)) == pytest.approx(0.0, abs=1e-9)

    def test_ninety_degree_yaw(self):
        a = quat_identity()
        b = quat_from_euler(0.0, 0.0, 90 * DEG)
        assert angular_distance(a, b) == pytest.approx(90 * DEG, abs=1e-9)

    def test_never_exceeds_pi(self):
        a = quat_identity()
        b = quat_from_euler(0.0, 0.0, 179 * DEG)
        assert 0.0 <= angular_distance(a, b) <= math.pi + 1e-12
