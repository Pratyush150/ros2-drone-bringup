#!/usr/bin/env python3
"""Aggregate MAVROS telemetry into a single go/no-go preflight verdict.

The verdict is published two ways:

* ``diagnostic_msgs/DiagnosticArray`` on ``/diagnostics``, one status per check,
  so it appears in ``rqt_runtime_monitor`` and in any diagnostic aggregator you
  already run.
* ``std_msgs/Bool`` on ``~/ready``, latched (TRANSIENT_LOCAL), so the mission
  executor can gate arming on it and a node that starts late still gets the
  current answer instead of waiting for the next change.

All the actual decision logic lives in
:func:`drone_bringup.core.state_machine.check_preflight`, which is pure Python
and unit tested. This node only converts messages into a
:class:`~drone_bringup.core.state_machine.VehicleSnapshot` and publishes the
result. Keeping it that thin is the point -- the thing that decides whether a
vehicle may arm should be testable without a middleware.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import GPSRAW, HomePosition, RCIn, State
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, NavSatFix
from std_msgs.msg import Bool, String

from drone_bringup.core.state_machine import (
    PreflightLimits,
    PreflightStatus,
    VehicleSnapshot,
    check_preflight,
)
from drone_bringup.nodes.qos import command_qos, sensor_qos, state_qos


class PreflightCheckNode(Node):
    """Publishes a preflight verdict at a fixed rate."""

    def __init__(self) -> None:
        super().__init__("preflight_check")

        self._declare_parameters()
        self._limits = self._limits_from_parameters()

        ns = str(self.get_parameter("mavros_namespace").value).rstrip("/")
        self._geofence_topic = str(self.get_parameter("geofence_status_topic").value)

        self._state: Optional[State] = None
        self._fix: Optional[NavSatFix] = None
        self._fix_t = 0.0
        self._battery: Optional[BatteryState] = None
        self._pose: Optional[PoseStamped] = None
        self._pose_t = 0.0
        self._rc: Optional[RCIn] = None
        self._rc_t = 0.0
        self._gps_raw: Optional[GPSRAW] = None
        self._home_set = False
        self._geofence_loaded = False

        self.create_subscription(State, f"{ns}/state", self._on_state, state_qos())
        self.create_subscription(
            NavSatFix, f"{ns}/global_position/global", self._on_fix, sensor_qos()
        )
        self.create_subscription(
            BatteryState, f"{ns}/battery", self._on_battery, sensor_qos()
        )
        self.create_subscription(
            PoseStamped, f"{ns}/local_position/pose", self._on_pose, sensor_qos()
        )
        self.create_subscription(RCIn, f"{ns}/rc/in", self._on_rc, sensor_qos())
        # GPSRAW carries the real MAVLink fix type, satellite count and eph
        # (HDOP * 100). NavSatFix alone cannot tell you any of those.
        self.create_subscription(
            GPSRAW, f"{ns}/gpsstatus/gps1/raw", self._on_gps_raw, sensor_qos()
        )
        self.create_subscription(
            HomePosition, f"{ns}/home_position/home", self._on_home, state_qos()
        )
        # The geofence monitor latches its load status; TRANSIENT_LOCAL means we
        # get the current value even if we started after it did.
        self.create_subscription(
            String, self._geofence_topic, self._on_geofence, state_qos()
        )

        self._pub_ready = self.create_publisher(Bool, "~/ready", state_qos())
        self._pub_report = self.create_publisher(String, "~/report", state_qos())
        self._pub_diag = self.create_publisher(
            DiagnosticArray, "/diagnostics", command_qos()
        )

        rate = float(self.get_parameter("check_rate_hz").value)
        self._timer = self.create_timer(1.0 / max(1e-3, rate), self._on_timer)
        self._last_passed: Optional[bool] = None

        self.get_logger().info(f"preflight_check up, watching {ns}")

    # -- parameters ---------------------------------------------------------
    def _declare_parameters(self) -> None:
        """Declare thresholds as parameters so they can be tuned per airframe."""
        self.declare_parameter(
            "mavros_namespace",
            "/mavros",
            ParameterDescriptor(description="MAVROS namespace.", read_only=True),
        )
        self.declare_parameter(
            "geofence_status_topic",
            "/geofence_monitor/status",
            ParameterDescriptor(
                description="Latched topic the geofence monitor publishes its "
                "load status on.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "check_rate_hz",
            2.0,
            ParameterDescriptor(
                description="How often the verdict is recomputed and published.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.1, to_value=50.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "min_gps_fix_type",
            3,
            ParameterDescriptor(
                description="Minimum MAVLink GPS_FIX_TYPE. 3 = 3D fix. Below 3 "
                "there is no usable altitude.",
                integer_range=[IntegerRange(from_value=0, to_value=8, step=1)],
            ),
        )
        self.declare_parameter(
            "min_satellites",
            8,
            ParameterDescriptor(
                description="Minimum satellite count.",
                integer_range=[IntegerRange(from_value=0, to_value=40, step=1)],
            ),
        )
        self.declare_parameter(
            "min_battery_voltage",
            14.0,
            ParameterDescriptor(
                description="Minimum pack voltage in volts. Set this for YOUR "
                "cell count: 14.0 V suits a 4S pack, not a 6S one.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.0, to_value=60.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "min_battery_percent",
            0.40,
            ParameterDescriptor(
                description="Minimum state of charge as a fraction in [0, 1].",
                floating_point_range=[
                    FloatingPointRange(from_value=0.0, to_value=1.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "max_hdop",
            2.0,
            ParameterDescriptor(
                description="Maximum acceptable horizontal dilution of precision.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.1, to_value=50.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "max_telemetry_age_s",
            1.0,
            ParameterDescriptor(
                description="Inputs older than this fail preflight outright.",
                floating_point_range=[
                    FloatingPointRange(from_value=0.05, to_value=30.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "require_rc",
            True,
            ParameterDescriptor(
                description="Require a live RC link. Turn this off only for SITL "
                "or for a vehicle with no safety pilot, and know why."
            ),
        )
        self.declare_parameter(
            "require_geofence",
            True,
            ParameterDescriptor(description="Require a loaded geofence before arming."),
        )
        self.declare_parameter(
            "require_home_set",
            True,
            ParameterDescriptor(
                description="Require a home position; without one RTL has nowhere to go."
            ),
        )

    def _limits_from_parameters(self) -> PreflightLimits:
        """Snapshot the parameters into an immutable limits object."""
        get = self.get_parameter
        return PreflightLimits(
            min_gps_fix_type=int(get("min_gps_fix_type").value),
            min_satellites=int(get("min_satellites").value),
            min_battery_voltage=float(get("min_battery_voltage").value),
            min_battery_percent=float(get("min_battery_percent").value),
            max_hdop=float(get("max_hdop").value),
            max_telemetry_age_s=float(get("max_telemetry_age_s").value),
            require_rc=bool(get("require_rc").value),
            require_geofence=bool(get("require_geofence").value),
            require_home_set=bool(get("require_home_set").value),
        )

    # -- callbacks ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg: State) -> None:
        self._state = msg

    def _on_fix(self, msg: NavSatFix) -> None:
        self._fix = msg
        self._fix_t = self._now()

    def _on_battery(self, msg: BatteryState) -> None:
        self._battery = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg
        self._pose_t = self._now()

    def _on_rc(self, msg: RCIn) -> None:
        self._rc = msg
        self._rc_t = self._now()

    def _on_gps_raw(self, msg: GPSRAW) -> None:
        self._gps_raw = msg

    def _on_home(self, msg: HomePosition) -> None:
        # A HomePosition message at all means the FCU has set home. The values
        # are only meaningful once it has.
        self._home_set = True
        self.get_logger().info(
            f"home set: {msg.geo.latitude:.7f}, {msg.geo.longitude:.7f}"
        )

    def _on_geofence(self, msg: String) -> None:
        self._geofence_loaded = msg.data.startswith("loaded")

    # -- snapshot -----------------------------------------------------------
    def build_snapshot(self) -> VehicleSnapshot:
        """Turn the latest messages into a pure-Python snapshot.

        Anything we have never received maps to the "unsafe" value, so a missing
        subscription fails preflight rather than passing it by omission.
        """
        now = self._now()
        # Prefer GPSRAW: it is the only place the real MAVLink fix type,
        # satellite count and eph (HDOP * 100) are exposed. Fall back to the
        # coarse NavSatFix status if GPSRAW is not being published.
        fix_type = 0
        satellites = 0
        hdop = 99.0
        if self._gps_raw is not None:
            fix_type = int(self._gps_raw.fix_type)
            satellites = int(self._gps_raw.satellites_visible)
            eph = int(self._gps_raw.eph)
            # eph is UINT16_MAX when unknown.
            hdop = 99.0 if eph in (0, 65535) else eph / 100.0
        elif self._fix is not None:
            fix_type = 3 if self._fix.status.status >= 0 else 0

        rc_connected = self._rc is not None and (
            now - self._rc_t
        ) <= self._limits.max_telemetry_age_s

        telemetry_age = max(
            now - self._fix_t if self._fix_t > 0.0 else 99.0,
            now - self._pose_t if self._pose_t > 0.0 else 99.0,
        )

        return VehicleSnapshot(
            gps_fix_type=fix_type,
            satellites=satellites,
            hdop=hdop,
            ekf_ok=self._pose is not None
            and (now - self._pose_t) <= self._limits.max_telemetry_age_s,
            battery_voltage=0.0 if self._battery is None else float(self._battery.voltage),
            battery_percent=0.0
            if self._battery is None
            else float(self._battery.percentage),
            rc_connected=rc_connected,
            geofence_loaded=self._geofence_loaded,
            home_set=self._home_set,
            armed=self._state is not None and bool(self._state.armed),
            telemetry_age_s=telemetry_age,
            altitude_m=0.0 if self._pose is None else float(self._pose.pose.position.z),
        )

    # -- periodic work ------------------------------------------------------
    def _on_timer(self) -> None:
        snapshot = self.build_snapshot()
        status = check_preflight(snapshot, self._limits)

        ready = Bool()
        ready.data = status.passed
        self._pub_ready.publish(ready)

        report = String()
        report.data = status.summary()
        self._pub_report.publish(report)

        self._pub_diag.publish(self._to_diagnostics(snapshot, status))

        # Log only on change: a 2 Hz log line for a vehicle sitting on the bench
        # buries everything else.
        if status.passed != self._last_passed:
            if status.passed:
                self.get_logger().info("preflight PASS")
            else:
                self.get_logger().warn(status.summary())
            self._last_passed = status.passed

    def _to_diagnostics(
        self, snapshot: VehicleSnapshot, status: PreflightStatus
    ) -> DiagnosticArray:
        """Render the verdict as a DiagnosticArray with one aggregate status."""
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        item = DiagnosticStatus(
            name="drone_bringup: preflight",
            hardware_id="fcu",
            level=DiagnosticStatus.OK if status.passed else DiagnosticStatus.ERROR,
            message=status.summary(),
        )
        item.values = [
            KeyValue(key="gps_fix_type", value=str(snapshot.gps_fix_type)),
            KeyValue(key="satellites", value=str(snapshot.satellites)),
            KeyValue(key="hdop", value=f"{snapshot.hdop:.2f}"),
            KeyValue(key="ekf_ok", value=str(snapshot.ekf_ok)),
            KeyValue(key="battery_v", value=f"{snapshot.battery_voltage:.2f}"),
            KeyValue(key="battery_pct", value=f"{snapshot.battery_percent:.2f}"),
            KeyValue(key="rc_connected", value=str(snapshot.rc_connected)),
            KeyValue(key="geofence_loaded", value=str(snapshot.geofence_loaded)),
            KeyValue(key="telemetry_age_s", value=f"{snapshot.telemetry_age_s:.2f}"),
            KeyValue(key="failures", value=str(len(status.failures))),
        ]
        array.status = [item]
        return array

    def destroy_node(self) -> bool:
        """Cancel the timer before publishers are torn down."""
        if self._timer is not None:
            self._timer.cancel()
        return super().destroy_node()


def main(args=None) -> None:
    """Console-script entry point."""
    rclpy.init(args=args)
    node = PreflightCheckNode()
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
