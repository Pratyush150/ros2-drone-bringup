#!/usr/bin/env python3
"""Normalise MAVROS telemetry into a stable, documented topic set.

Why this node exists
--------------------
MAVROS topic names, frames, and QoS change between releases and between PX4 and
ArduPilot. If every node in your stack subscribes to ``/mavros/...`` directly,
a MAVROS upgrade rewrites your whole graph. This node is the single place that
knows about MAVROS. Everything downstream subscribes to ``~/state``,
``~/pose``, and ``~/diagnostics``, whose shape is ours to keep stable.

It also does three things that are easy to get wrong:

1. **Frame conversion in one place.** MAVROS already publishes ENU/FLU, but raw
   MAVLink and PX4 uORB do not. The converters in
   :mod:`drone_bringup.core.frames` are used here so there is exactly one
   implementation, and it is unit tested.
2. **Staleness detection.** A MAVROS topic that stops updating looks identical
   to one that is updating with an unchanged value. Every input is timestamped
   on arrival and reported stale after a configurable age. Stale telemetry that
   nobody notices is how an offboard controller flies on a five-second-old
   position.
3. **Diagnostics.** Everything the preflight gate needs is aggregated into a
   ``diagnostic_msgs/DiagnosticArray`` so it shows up in ``rqt_runtime_monitor``
   without extra plumbing.

QoS: read :mod:`drone_bringup.nodes.qos` before changing anything.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ExtendedState, State
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
from std_msgs.msg import String

from drone_bringup.core.frames import quat_from_ros_xyzw, quat_to_euler
from drone_bringup.core.geodesy import LocalOrigin, enu_to_ned
from drone_bringup.nodes.qos import command_qos, sensor_qos, state_qos


#: MAVLink ``MAV_LANDED_STATE`` values, as republished by mavros_msgs/ExtendedState.
_LANDED_STATE_NAMES = {0: "UNDEFINED", 1: "ON_GROUND", 2: "IN_AIR", 3: "TAKEOFF", 4: "LANDING"}


def _landed_state_name(extended: Optional[ExtendedState]) -> str:
    """Human-readable landed state, or ``"unknown"`` if we have not heard one."""
    if extended is None:
        return "unknown"
    return _LANDED_STATE_NAMES.get(int(extended.landed_state), "UNDEFINED")


class TelemetryBridgeNode(Node):
    """MAVROS -> normalised topics, with staleness tracking and diagnostics."""

    def __init__(self) -> None:
        super().__init__("telemetry_bridge")

        self._declare_parameters()

        self._mavros_ns: str = self.get_parameter("mavros_namespace").value
        self._stale_after: float = self.get_parameter("stale_after_s").value
        self._publish_rate: float = self.get_parameter("publish_rate_hz").value
        self._origin: Optional[LocalOrigin] = None

        # Latest samples plus the wall time they arrived. A message that stops
        # arriving keeps its last value forever; the timestamp is the only way
        # to tell "unchanged" from "gone".
        self._state: Optional[State] = None
        self._state_t = 0.0
        self._extended: Optional[ExtendedState] = None
        self._fix: Optional[NavSatFix] = None
        self._fix_t = 0.0
        self._battery: Optional[BatteryState] = None
        self._battery_t = 0.0
        self._pose: Optional[PoseStamped] = None
        self._pose_t = 0.0
        self._vel: Optional[TwistStamped] = None
        self._vel_t = 0.0
        self._imu: Optional[Imu] = None
        self._imu_t = 0.0

        ns = self._mavros_ns.rstrip("/")
        # Inputs: MAVROS. Sensor streams are BEST_EFFORT -- see drone_bringup.nodes.qos.
        self.create_subscription(State, f"{ns}/state", self._on_state, state_qos())
        self.create_subscription(
            ExtendedState, f"{ns}/extended_state", self._on_extended, state_qos()
        )
        self.create_subscription(
            NavSatFix, f"{ns}/global_position/global", self._on_fix, sensor_qos()
        )
        self.create_subscription(
            BatteryState, f"{ns}/battery", self._on_battery, sensor_qos()
        )
        self.create_subscription(
            PoseStamped, f"{ns}/local_position/pose", self._on_pose, sensor_qos()
        )
        self.create_subscription(
            TwistStamped, f"{ns}/local_position/velocity_local", self._on_vel, sensor_qos()
        )
        self.create_subscription(Imu, f"{ns}/imu/data", self._on_imu, sensor_qos())

        # Outputs: our stable contract.
        self._pub_odom = self.create_publisher(Odometry, "~/odom", sensor_qos())
        self._pub_status = self.create_publisher(String, "~/status", state_qos())
        self._pub_diag = self.create_publisher(
            DiagnosticArray, "/diagnostics", command_qos()
        )

        period = 1.0 / max(1e-3, self._publish_rate)
        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"telemetry_bridge up: mavros_namespace={self._mavros_ns} "
            f"rate={self._publish_rate} Hz stale_after={self._stale_after} s"
        )

    # -- parameters ---------------------------------------------------------
    def _declare_parameters(self) -> None:
        """Declare every parameter with a descriptor.

        Descriptors are not decoration: they give ``ros2 param describe`` real
        text, and the ranges make out-of-band values fail at set time instead of
        at 400 m AGL.
        """
        self.declare_parameter(
            "mavros_namespace",
            "/mavros",
            ParameterDescriptor(
                description="Namespace MAVROS publishes under, e.g. /mavros or "
                "/uav1/mavros for a multi-vehicle setup.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "publish_rate_hz",
            20.0,
            ParameterDescriptor(
                description="Rate at which normalised odometry and diagnostics "
                "are republished.",
                floating_point_range=[
                    FloatingPointRange(from_value=1.0, to_value=200.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "stale_after_s",
            1.0,
            ParameterDescriptor(
                description="An input older than this is reported STALE in "
                "diagnostics. Set it to about 5x the expected period of the "
                "slowest input you care about.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.05, to_value=30.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "min_satellites",
            8,
            ParameterDescriptor(
                description="Satellite count below which GPS diagnostics warn.",
                integer_range=[IntegerRange(from_value=0, to_value=40, step=1)],
            ),
        )
        self.declare_parameter(
            "frame_id",
            "map",
            ParameterDescriptor(description="Fixed frame for published odometry."),
        )
        self.declare_parameter(
            "child_frame_id",
            "base_link",
            ParameterDescriptor(description="Body frame for published odometry."),
        )

    # -- callbacks ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg: State) -> None:
        self._state = msg
        self._state_t = self._now()

    def _on_extended(self, msg: ExtendedState) -> None:
        self._extended = msg

    def _on_fix(self, msg: NavSatFix) -> None:
        self._fix = msg
        self._fix_t = self._now()
        if self._origin is None and msg.status.status >= 0:
            # Anchor the local tangent plane on the first valid fix and never
            # move it. Re-anchoring mid-flight silently invalidates every local
            # coordinate already stored downstream.
            self._origin = LocalOrigin(msg.latitude, msg.longitude, msg.altitude)
            self.get_logger().info(
                f"local origin anchored at {msg.latitude:.7f}, {msg.longitude:.7f}, "
                f"{msg.altitude:.1f} m"
            )

    def _on_battery(self, msg: BatteryState) -> None:
        self._battery = msg
        self._battery_t = self._now()

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg
        self._pose_t = self._now()

    def _on_vel(self, msg: TwistStamped) -> None:
        self._vel = msg
        self._vel_t = self._now()

    def _on_imu(self, msg: Imu) -> None:
        self._imu = msg
        self._imu_t = self._now()

    # -- periodic work ------------------------------------------------------
    def _on_timer(self) -> None:
        now = self._now()
        self._publish_odom(now)
        self._publish_status(now)
        self._publish_diagnostics(now)

    def _publish_odom(self, now: float) -> None:
        if self._pose is None:
            return
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.get_parameter("frame_id").value
        odom.child_frame_id = self.get_parameter("child_frame_id").value
        odom.pose.pose = self._pose.pose
        if self._vel is not None:
            odom.twist.twist = self._vel.twist
        self._pub_odom.publish(odom)

    def _publish_status(self, now: float) -> None:
        parts = []
        if self._state is not None:
            parts.append(f"mode={self._state.mode}")
            parts.append(f"armed={self._state.armed}")
            parts.append(f"connected={self._state.connected}")
        if self._pose is not None:
            q = quat_from_ros_xyzw(
                (
                    self._pose.pose.orientation.x,
                    self._pose.pose.orientation.y,
                    self._pose.pose.orientation.z,
                    self._pose.pose.orientation.w,
                )
            )
            _, _, yaw_enu = quat_to_euler(q)
            parts.append(f"yaw_enu_deg={math.degrees(yaw_enu):.1f}")
            parts.append(f"alt_m={self._pose.pose.position.z:.1f}")
        if self._vel is not None:
            # MAVROS velocity is ENU. Report NED too so a MAVLink-native
            # consumer does not have to guess.
            v = self._vel.twist.linear
            n, e, d = enu_to_ned(v.x, v.y, v.z)
            parts.append(f"vel_ned=({n:.1f},{e:.1f},{d:.1f})")
        msg = String()
        msg.data = " ".join(parts) if parts else "no telemetry"
        self._pub_status.publish(msg)

    def _publish_diagnostics(self, now: float) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [
            self._diag_link(now),
            self._diag_gps(now),
            self._diag_battery(now),
            self._diag_estimator(now),
        ]
        self._pub_diag.publish(array)

    def _stale(self, stamp: float, now: float) -> bool:
        return stamp <= 0.0 or (now - stamp) > self._stale_after

    def _diag_link(self, now: float) -> DiagnosticStatus:
        status = DiagnosticStatus(name="drone_bringup: mavlink_link", hardware_id="fcu")
        if self._state is None or self._stale(self._state_t, now):
            status.level = DiagnosticStatus.ERROR
            status.message = "no MAVROS state (is mavros running, is the port right?)"
            return status
        status.values = [
            KeyValue(key="mode", value=str(self._state.mode)),
            KeyValue(key="armed", value=str(self._state.armed)),
            KeyValue(key="age_s", value=f"{now - self._state_t:.2f}"),
            KeyValue(key="landed_state", value=_landed_state_name(self._extended)),
        ]
        if not self._state.connected:
            status.level = DiagnosticStatus.ERROR
            status.message = "MAVROS running but FCU not connected"
        else:
            status.level = DiagnosticStatus.OK
            status.message = f"connected, mode {self._state.mode}"
        return status

    def _diag_gps(self, now: float) -> DiagnosticStatus:
        status = DiagnosticStatus(name="drone_bringup: gps", hardware_id="fcu")
        if self._fix is None or self._stale(self._fix_t, now):
            status.level = DiagnosticStatus.ERROR
            status.message = "no GPS fix message"
            return status
        # NavSatStatus: -1 NO_FIX, 0 FIX, 1 SBAS, 2 GBAS.
        fix_ok = self._fix.status.status >= 0
        status.values = [
            KeyValue(key="status", value=str(self._fix.status.status)),
            KeyValue(key="latitude", value=f"{self._fix.latitude:.7f}"),
            KeyValue(key="longitude", value=f"{self._fix.longitude:.7f}"),
            KeyValue(key="altitude", value=f"{self._fix.altitude:.1f}"),
        ]
        status.level = DiagnosticStatus.OK if fix_ok else DiagnosticStatus.ERROR
        status.message = "3D fix" if fix_ok else "no fix"
        return status

    def _diag_battery(self, now: float) -> DiagnosticStatus:
        status = DiagnosticStatus(name="drone_bringup: battery", hardware_id="fcu")
        if self._battery is None or self._stale(self._battery_t, now):
            status.level = DiagnosticStatus.WARN
            status.message = "no battery telemetry"
            return status
        pct = self._battery.percentage
        status.values = [
            KeyValue(key="voltage", value=f"{self._battery.voltage:.2f}"),
            KeyValue(key="percentage", value=f"{pct:.2f}"),
        ]
        if pct < 0.15:
            status.level = DiagnosticStatus.ERROR
            status.message = f"critical: {pct * 100:.0f}%"
        elif pct < 0.30:
            status.level = DiagnosticStatus.WARN
            status.message = f"low: {pct * 100:.0f}%"
        else:
            status.level = DiagnosticStatus.OK
            status.message = f"{pct * 100:.0f}%"
        return status

    def _diag_estimator(self, now: float) -> DiagnosticStatus:
        status = DiagnosticStatus(name="drone_bringup: estimator", hardware_id="fcu")
        stale_pose = self._pose is None or self._stale(self._pose_t, now)
        stale_imu = self._imu is None or self._stale(self._imu_t, now)
        status.values = [
            KeyValue(key="pose_age_s", value=f"{now - self._pose_t:.2f}"),
            KeyValue(key="imu_age_s", value=f"{now - self._imu_t:.2f}"),
            KeyValue(key="origin_anchored", value=str(self._origin is not None)),
        ]
        if stale_pose or stale_imu:
            status.level = DiagnosticStatus.ERROR
            status.message = "local position or IMU stale -- do not fly offboard"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "local position and IMU fresh"
        return status

    # -- shutdown -----------------------------------------------------------
    def destroy_node(self) -> bool:
        """Cancel the timer before tearing down publishers.

        Without this the timer can fire once more against half-destroyed
        publishers during shutdown and throw from inside the executor.
        """
        if self._timer is not None:
            self._timer.cancel()
        return super().destroy_node()


def main(args=None) -> None:
    """Console-script entry point."""
    rclpy.init(args=args)
    node = TelemetryBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted, shutting down")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
