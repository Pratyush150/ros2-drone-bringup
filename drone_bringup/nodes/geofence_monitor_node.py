#!/usr/bin/env python3
"""Watch position and velocity against a loaded geofence; warn before the breach.

The interesting output is not "you are outside the fence" -- by then you are
already outside it. It is ``time_to_breach``, computed from the current
velocity. A 15 m/s multirotor needs several seconds and tens of metres to stop,
so the useful trigger is "you will cross the boundary in under N seconds".

Outputs
-------
``~/status``  ``std_msgs/String``, latched. Starts as ``loaded: <n> zones`` or
              ``error: <why>``. The preflight node gates on this, which is why
              it is TRANSIENT_LOCAL: a preflight node that starts later still
              learns the fence was loaded.
``~/breach``  ``std_msgs/Bool``, latched. True while the fence is violated.
``~/margin``  ``std_msgs/Float32``. Metres of margin; negative when breached.
``~/markers`` ``visualization_msgs/MarkerArray``, latched, for RViz.
``/diagnostics``

This node **does not command the vehicle**. It publishes a verdict; the mission
executor decides what to do about it. Splitting those two apart means you can
run the monitor in "advisory" mode on a manually flown vehicle without it ever
touching the control path.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, ColorRGBA, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from drone_bringup.core.geofence import Geofence, GeofenceError, ZoneKind
from drone_bringup.nodes.qos import command_qos, sensor_qos, state_qos


class GeofenceMonitorNode(Node):
    """Loads a geofence YAML and evaluates the vehicle against it."""

    def __init__(self) -> None:
        super().__init__("geofence_monitor")

        self._declare_parameters()

        self._horizon_s = float(self.get_parameter("prediction_horizon_s").value)
        self._warn_time_s = float(self.get_parameter("warn_time_to_breach_s").value)
        self._warn_margin_m = float(self.get_parameter("warn_margin_m").value)

        self._fence: Optional[Geofence] = None
        self._load_error: Optional[str] = None
        self._fix: Optional[NavSatFix] = None
        self._fix_t = 0.0
        self._vel: Optional[TwistStamped] = None
        self._pose: Optional[PoseStamped] = None

        ns = str(self.get_parameter("mavros_namespace").value).rstrip("/")
        self.create_subscription(
            NavSatFix, f"{ns}/global_position/global", self._on_fix, sensor_qos()
        )
        self.create_subscription(
            TwistStamped, f"{ns}/local_position/velocity_local", self._on_vel, sensor_qos()
        )
        self.create_subscription(
            PoseStamped, f"{ns}/local_position/pose", self._on_pose, sensor_qos()
        )

        self._pub_status = self.create_publisher(String, "~/status", state_qos())
        self._pub_breach = self.create_publisher(Bool, "~/breach", state_qos())
        self._pub_margin = self.create_publisher(Float32, "~/margin", sensor_qos())
        self._pub_markers = self.create_publisher(MarkerArray, "~/markers", state_qos())
        self._pub_diag = self.create_publisher(
            DiagnosticArray, "/diagnostics", command_qos()
        )

        self._load_fence(str(self.get_parameter("geofence_file").value))

        rate = float(self.get_parameter("check_rate_hz").value)
        self._timer = self.create_timer(1.0 / max(1e-3, rate), self._on_timer)
        self._last_breach: Optional[bool] = None

    # -- parameters ---------------------------------------------------------
    def _declare_parameters(self) -> None:
        """Declare parameters with descriptors and ranges."""
        self.declare_parameter(
            "mavros_namespace",
            "/mavros",
            ParameterDescriptor(description="MAVROS namespace.", read_only=True),
        )
        self.declare_parameter(
            "geofence_file",
            "",
            ParameterDescriptor(
                description="Absolute path to the geofence YAML. Empty means no "
                "fence is loaded and the monitor reports an error, which fails "
                "preflight -- that is deliberate.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "check_rate_hz",
            10.0,
            ParameterDescriptor(
                description="Evaluation rate. Should be at least as fast as your "
                "position update rate.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.5, to_value=100.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "prediction_horizon_s",
            30.0,
            ParameterDescriptor(
                description="Predicted breaches further out than this are ignored.",
                floating_point_range=[
                    FloatingPointRange(from_value=1.0, to_value=300.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "warn_time_to_breach_s",
            5.0,
            ParameterDescriptor(
                description="Raise a WARN diagnostic when the predicted breach is "
                "closer than this. Set it to your worst-case stopping time plus "
                "your command latency, not to a round number.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.5, to_value=60.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "warn_margin_m",
            10.0,
            ParameterDescriptor(
                description="Raise a WARN diagnostic when the boundary margin "
                "drops below this many metres.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.0, to_value=1000.0, step=0.0)
                ],
            ),
        )

    # -- fence loading ------------------------------------------------------
    def _load_fence(self, path: str) -> None:
        """Load and validate the geofence file, latching the outcome on ``~/status``."""
        status = String()
        if not path:
            self._load_error = "no geofence_file parameter set"
        elif not os.path.isfile(path):
            self._load_error = f"geofence file not found: {path}"
        else:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle)
                self._fence = Geofence.from_dict(data)
                self._load_error = None
            except (OSError, yaml.YAMLError, GeofenceError) as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"

        if self._fence is not None:
            status.data = (
                f"loaded: {len(self._fence.zones)} zones "
                f"({len(self._fence.inclusion_zones)} inclusion, "
                f"{len(self._fence.exclusion_zones)} exclusion), "
                f"ceiling={self._fence.max_altitude_m}"
            )
            self.get_logger().info(status.data)
            self._pub_markers.publish(self._build_markers())
        else:
            status.data = f"error: {self._load_error}"
            self.get_logger().error(status.data)
        self._pub_status.publish(status)

    # -- callbacks ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_fix(self, msg: NavSatFix) -> None:
        self._fix = msg
        self._fix_t = self._now()

    def _on_vel(self, msg: TwistStamped) -> None:
        self._vel = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg

    # -- periodic work ------------------------------------------------------
    def _on_timer(self) -> None:
        if self._fence is None:
            self._publish_diag_error(self._load_error or "no fence")
            return
        if self._fix is None:
            self._publish_diag_error("no GPS fix yet")
            return

        # MAVROS velocity_local is ENU: x East, y North, z Up.
        vel_e = vel_n = vel_u = 0.0
        if self._vel is not None:
            vel_e = float(self._vel.twist.linear.x)
            vel_n = float(self._vel.twist.linear.y)
            vel_u = float(self._vel.twist.linear.z)

        # Horizontal position comes from the GPS fix projected through the
        # fence origin. Altitude prefers the MAVROS local pose, because that is
        # already metres-above-origin -- the same datum the fence uses. The raw
        # NavSatFix altitude is ellipsoidal height, so using it directly would
        # compare a WGS84 height against a local ceiling and be tens of metres
        # out in either direction.
        east, north, up_from_gps = self._fence.origin.geodetic_to_enu(
            self._fix.latitude, self._fix.longitude, self._fix.altitude
        )
        up = (
            float(self._pose.pose.position.z)
            if self._pose is not None
            else up_from_gps
        )
        status = self._fence.check_local(
            east,
            north,
            up,
            vel_e,
            vel_n,
            vel_u,
            horizon_s=self._horizon_s,
        )

        breach = Bool()
        breach.data = status.breached
        self._pub_breach.publish(breach)

        margin = Float32()
        margin.data = float(
            status.margin_m if math.isfinite(status.margin_m) else 1e6
        )
        self._pub_margin.publish(margin)

        self._pub_diag.publish(self._to_diagnostics(status))

        if status.breached != self._last_breach:
            if status.breached:
                self.get_logger().error(
                    "GEOFENCE BREACH: " + "; ".join(status.violations)
                )
            else:
                self.get_logger().info("geofence clear")
            self._last_breach = status.breached

    def _to_diagnostics(self, status) -> DiagnosticArray:
        """Render a geofence status as a DiagnosticArray."""
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        item = DiagnosticStatus(name="drone_bringup: geofence", hardware_id="fcu")
        ttb = (
            status.predicted_breach.time_to_breach_s
            if status.predicted_breach is not None
            else float("inf")
        )
        item.values = [
            KeyValue(key="margin_m", value=f"{status.margin_m:.1f}"),
            KeyValue(key="time_to_breach_s", value=f"{ttb:.1f}"),
            KeyValue(
                key="predicted",
                value=""
                if status.predicted_breach is None
                else str(status.predicted_breach),
            ),
            KeyValue(key="violations", value="; ".join(status.violations)),
        ]
        if status.breached:
            item.level = DiagnosticStatus.ERROR
            item.message = "; ".join(status.violations)
        elif ttb < self._warn_time_s or status.margin_m < self._warn_margin_m:
            item.level = DiagnosticStatus.WARN
            item.message = (
                f"approaching boundary: {status.margin_m:.0f} m margin, "
                f"{ttb:.1f} s to breach"
            )
        else:
            item.level = DiagnosticStatus.OK
            item.message = f"inside fence, {status.margin_m:.0f} m margin"
        array.status = [item]
        return array

    def _publish_diag_error(self, message: str) -> None:
        """Publish a single ERROR diagnostic when we cannot evaluate the fence."""
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [
            DiagnosticStatus(
                name="drone_bringup: geofence",
                hardware_id="fcu",
                level=DiagnosticStatus.ERROR,
                message=message,
            )
        ]
        self._pub_diag.publish(array)

    # -- visualisation ------------------------------------------------------
    def _build_markers(self) -> MarkerArray:
        """One LINE_STRIP per zone, in the local ``map`` frame."""
        array = MarkerArray()
        if self._fence is None:
            return array
        for i, zone in enumerate(self._fence.zones):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "geofence"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 2.0
            marker.pose.orientation.w = 1.0
            inclusion = zone.kind is ZoneKind.INCLUSION
            marker.color = ColorRGBA(
                r=0.1 if inclusion else 0.9,
                g=0.8 if inclusion else 0.2,
                b=0.2,
                a=0.9,
            )
            z = zone.max_alt_m if zone.max_alt_m is not None else 0.0
            points: List[Point] = [
                Point(x=float(e), y=float(n), z=float(z))
                for e, n in zone.polygon.vertices
            ]
            if points:
                points.append(points[0])  # close the ring
            marker.points = points
            array.markers.append(marker)
        return array

    def destroy_node(self) -> bool:
        """Cancel the timer before publishers are torn down."""
        if self._timer is not None:
            self._timer.cancel()
        return super().destroy_node()


def main(args=None) -> None:
    """Console-script entry point."""
    rclpy.init(args=args)
    node = GeofenceMonitorNode()
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
