#!/usr/bin/env python3
"""Drive a mission through the guarded state machine, talking to MAVROS.

Structure
---------
This node is deliberately thin. It owns:

* the MAVROS subscriptions and service clients,
* a fixed-rate tick that feeds the state machine and sends setpoints,
* the mapping from :class:`~drone_bringup.core.state_machine.MissionState` to
  the MAVROS call that implements it.

Everything else -- parsing the mission, expanding the survey grid, deciding
whether preflight passed, deciding whether a transition is legal -- lives in
:mod:`drone_bringup.core` and is unit tested without ROS.

Lifecycle-friendly, deliberately
--------------------------------
The node is written so it can be converted to a ``rclpy.lifecycle.LifecycleNode``
by moving the body of :meth:`configure` into ``on_configure`` and the body of
:meth:`activate` into ``on_activate``. Construction only declares parameters and
allocates state; nothing is published and no service is called until
:meth:`activate` runs. That split is what makes a lifecycle conversion a
refactor instead of a rewrite, and it is why the mission file is parsed in
:meth:`configure` -- a bad mission file should fail before anything can arm.

Offboard setpoint streaming
---------------------------
PX4 will drop out of OFFBOARD if setpoints stop arriving for ~0.5 s, and it will
refuse to *enter* OFFBOARD unless setpoints have already been streaming for a
couple of seconds. That is not a bug to work around; it is the failsafe working.
The tick therefore publishes a setpoint every cycle regardless of state, and
:attr:`_setpoint_stream_started` records when streaming began so we do not
request the mode change too early.
"""

from __future__ import annotations

import math
from typing import List, Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Path
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, String

from drone_bringup.core.frames import quat_from_euler, quat_to_ros_xyzw
from drone_bringup.core.geodesy import LocalOrigin, ned_yaw_to_enu_yaw
from drone_bringup.core.mission import Mission, MissionValidationError, Waypoint, load_mission_file
from drone_bringup.core.state_machine import (
    AbortReason,
    IllegalTransition,
    MissionState,
    MissionStateMachine,
)
from drone_bringup.nodes.qos import command_qos, sensor_qos, state_qos

#: PX4 requires a setpoint stream to be running before it will accept OFFBOARD.
#: Two seconds at 20 Hz is comfortably past the threshold on every build tested.
OFFBOARD_WARMUP_S = 2.0


class MissionExecutorNode(Node):
    """Executes a YAML mission plan against MAVROS."""

    def __init__(self) -> None:
        super().__init__("mission_executor")

        self._declare_parameters()

        self._sm = MissionStateMachine()
        self._mission: Optional[Mission] = None
        self._waypoints: List[Waypoint] = []
        self._origin: Optional[LocalOrigin] = None
        self._state: Optional[State] = None
        self._pose: Optional[PoseStamped] = None
        self._fix: Optional[NavSatFix] = None
        self._preflight_ready = False
        self._geofence_breach = False
        self._setpoint_stream_started: Optional[float] = None
        self._waypoint_reached_at: Optional[float] = None
        self._activated = False
        self._timer = None
        self._site_checked = False
        self._site_mismatch = False
        self._max_site_offset_m = float(self.get_parameter("max_site_offset_m").value)

        self._sm.on_transition = lambda record: self.get_logger().info(
            f"state {record.from_state.value} -> {record.to_state.value}"
            f"{': ' + record.reason if record.reason else ''}"
        )

        self.configure()
        if bool(self.get_parameter("auto_activate").value):
            self.activate()

    # -- parameters ---------------------------------------------------------
    def _declare_parameters(self) -> None:
        """Declare every parameter with a descriptor and, where useful, a range."""
        self.declare_parameter(
            "mavros_namespace",
            "/mavros",
            ParameterDescriptor(description="MAVROS namespace.", read_only=True),
        )
        self.declare_parameter(
            "mission_file",
            "",
            ParameterDescriptor(
                description="Absolute path to the mission YAML. Parsed at "
                "configure time so a bad file fails on the ground.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "auto_activate",
            True,
            ParameterDescriptor(
                description="Start the tick immediately. Set false when driving "
                "this node from a lifecycle manager.",
                read_only=True,
            ),
        )
        self.declare_parameter(
            "auto_start",
            False,
            ParameterDescriptor(
                description="Begin the mission as soon as preflight passes, with "
                "no operator command. Leave this false on a real vehicle."
            ),
        )
        self.declare_parameter(
            "tick_rate_hz",
            20.0,
            ParameterDescriptor(
                description="Setpoint and state-machine rate. Must stay above "
                "PX4's ~2 Hz OFFBOARD timeout with margin; 20 Hz is the norm.",
                floating_point_range=[
                    FloatingPointRange(from_value=5.0, to_value=100.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "takeoff_altitude_m",
            10.0,
            ParameterDescriptor(
                description="Fallback takeoff altitude when the mission does not "
                "specify one.",
                floating_point_range=[
                    FloatingPointRange(from_value=1.0, to_value=200.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "waypoint_timeout_s",
            120.0,
            ParameterDescriptor(
                description="Abort if a single waypoint is not reached in this "
                "long. Catches a vehicle fighting wind it cannot beat.",
                floating_point_range=[
                    FloatingPointRange(from_value=5.0, to_value=3600.0, step=0.0)
                ],
            ),
        )
        self.declare_parameter(
            "abort_on_geofence_breach",
            True,
            ParameterDescriptor(
                description="Route to RTL when the geofence monitor reports a "
                "breach. Turn it off only if something else owns that response."
            ),
        )
        self.declare_parameter(
            "max_site_offset_m",
            5000.0,
            ParameterDescriptor(
                description="Refuse to auto-start if the vehicle is further than "
                "this from the mission origin. Catches the mission file you "
                "copied from last month's site and forgot to edit.",
                floating_point_range=[
                    FloatingPointRange(from_value=10.0, to_value=1000000.0, step=0.0)
                ],
            ),
        )

    # -- lifecycle-shaped setup --------------------------------------------
    def configure(self) -> bool:
        """Parse the mission and build every interface. No I/O starts here.

        Returns:
            True on success. On failure the node stays in IDLE with the reason
            logged; it will never arm.
        """
        path = str(self.get_parameter("mission_file").value)
        if path:
            try:
                self._mission = load_mission_file(path)
                self._waypoints = self._mission.expand()
                self._origin = self._mission.origin
                self._sm.begin_mission(len(self._waypoints))
                self.get_logger().info(
                    f"mission '{self._mission.name}' loaded: "
                    f"{len(self._mission.items)} items -> "
                    f"{len(self._waypoints)} waypoints, "
                    f"{self._mission.total_ground_distance():.0f} m, "
                    f"~{self._mission.estimated_duration_s() / 60.0:.1f} min"
                )
            except (OSError, MissionValidationError) as exc:
                self.get_logger().error(f"mission load failed: {exc}")
                self._mission = None
        else:
            self.get_logger().warn("no mission_file set; executor will idle")

        ns = str(self.get_parameter("mavros_namespace").value).rstrip("/")
        self.create_subscription(State, f"{ns}/state", self._on_state, state_qos())
        self.create_subscription(
            PoseStamped, f"{ns}/local_position/pose", self._on_pose, sensor_qos()
        )
        self.create_subscription(
            NavSatFix, f"{ns}/global_position/global", self._on_fix, sensor_qos()
        )
        self.create_subscription(
            Bool, "/preflight_check/ready", self._on_preflight, state_qos()
        )
        self.create_subscription(
            Bool, "/geofence_monitor/breach", self._on_geofence, state_qos()
        )
        self.create_subscription(String, "~/command", self._on_command, command_qos())

        self._pub_setpoint = self.create_publisher(
            PoseStamped, f"{ns}/setpoint_position/local", command_qos()
        )
        self._pub_state = self.create_publisher(String, "~/state", state_qos())
        # Latched: RViz almost always starts after the executor, and the path
        # is published exactly once. TRANSIENT_LOCAL is what makes it show up
        # anyway -- with VOLATILE the display stays empty forever.
        self._pub_path = self.create_publisher(Path, "~/path", state_qos())
        self._pub_diag = self.create_publisher(
            DiagnosticArray, "/diagnostics", command_qos()
        )

        self._cli_arm = self.create_client(CommandBool, f"{ns}/cmd/arming")
        self._cli_mode = self.create_client(SetMode, f"{ns}/set_mode")
        self._cli_takeoff = self.create_client(CommandTOL, f"{ns}/cmd/takeoff")
        self._cli_land = self.create_client(CommandTOL, f"{ns}/cmd/land")
        if self._waypoints:
            self._pub_path.publish(self._build_path())
        return self._mission is not None

    def _build_path(self) -> Path:
        """Render the expanded waypoint list as a nav_msgs/Path in local ENU."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        if self._mission is None:
            return path
        for waypoint in self._waypoints:
            east, north, up = self._mission.waypoint_to_enu(waypoint)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = east
            pose.pose.position.y = north
            pose.pose.position.z = up
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def activate(self) -> None:
        """Start the tick timer. Nothing is commanded before this runs."""
        if self._activated:
            return
        rate = float(self.get_parameter("tick_rate_hz").value)
        self._timer = self.create_timer(1.0 / max(1e-3, rate), self._tick)
        self._activated = True
        self.get_logger().info(f"mission_executor active at {rate:.0f} Hz")

    def deactivate(self) -> None:
        """Stop the tick. Setpoints stop, so PX4 will drop OFFBOARD by design."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._activated = False

    # -- callbacks ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg: State) -> None:
        self._state = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _on_fix(self, msg: NavSatFix) -> None:
        self._fix = msg
        if self._origin is not None and not self._site_checked:
            self._site_checked = True
            offset = self._origin.ground_distance_to(msg.latitude, msg.longitude)
            if offset > self._max_site_offset_m:
                self.get_logger().error(
                    f"vehicle is {offset / 1000.0:.1f} km from the mission origin; "
                    f"this mission was almost certainly written for a different "
                    f"site. Refusing to auto-start."
                )
                self._site_mismatch = True
            else:
                self.get_logger().info(
                    f"vehicle is {offset:.0f} m from the mission origin"
                )

    def _on_preflight(self, msg: Bool) -> None:
        self._preflight_ready = bool(msg.data)

    def _on_geofence(self, msg: Bool) -> None:
        breach = bool(msg.data)
        newly_breached = breach and not self._geofence_breach
        self._geofence_breach = breach
        if (
            newly_breached
            and bool(self.get_parameter("abort_on_geofence_breach").value)
            and self._sm.is_airborne
        ):
            self._safe_abort(AbortReason.GEOFENCE_BREACH, "geofence monitor reported breach")

    def _on_command(self, msg: String) -> None:
        """Operator commands on ``~/command``: ``start``, ``abort``, ``rtl``, ``reset``."""
        command = msg.data.strip().lower()
        if command == "start":
            self._start_requested()
        elif command == "abort":
            self._safe_abort(AbortReason.OPERATOR, "operator abort")
        elif command == "rtl":
            if not self._sm.try_request(MissionState.RTL, "operator RTL", self._now()):
                self.get_logger().warn(
                    f"RTL rejected from state {self._sm.state.value}"
                )
        elif command == "reset":
            if not self._sm.try_request(MissionState.IDLE, "operator reset", self._now()):
                self.get_logger().warn(
                    f"reset rejected from state {self._sm.state.value}"
                )
        else:
            self.get_logger().warn(
                f"unknown command '{command}'; expected start|abort|rtl|reset"
            )

    def _start_requested(self) -> None:
        """Enter PREFLIGHT if the machine allows it."""
        if self._mission is None:
            self.get_logger().error("cannot start: no valid mission loaded")
            return
        if self._site_mismatch:
            self.get_logger().error(
                "cannot start: vehicle is too far from the mission origin"
            )
            return
        if not self._sm.try_request(
            MissionState.PREFLIGHT, "operator start", self._now()
        ):
            self.get_logger().warn(
                f"start rejected from state {self._sm.state.value}; "
                f"allowed: {sorted(s.value for s in self._sm.allowed)}"
            )

    def _safe_abort(self, reason: AbortReason, detail: str) -> None:
        """Abort, tolerating the states where abort is not a legal transition."""
        try:
            self._sm.abort(reason, detail, self._now())
        except IllegalTransition as exc:
            self.get_logger().warn(f"abort not possible: {exc}")

    # -- the tick -----------------------------------------------------------
    def _tick(self) -> None:
        """One control cycle: stream a setpoint, then advance the state machine."""
        now = self._now()
        self._publish_setpoint(now)

        state = self._sm.state
        if state is MissionState.IDLE:
            if bool(self.get_parameter("auto_start").value) and self._preflight_ready:
                self._start_requested()
        elif state is MissionState.PREFLIGHT:
            self._tick_preflight(now)
        elif state is MissionState.ARMING:
            self._tick_arming(now)
        elif state is MissionState.TAKEOFF:
            self._tick_takeoff(now)
        elif state is MissionState.MISSION:
            self._tick_mission(now)
        elif state is MissionState.RTL:
            self._tick_rtl(now)
        elif state is MissionState.LANDING:
            self._tick_landing(now)

        self._publish_state()
        self._publish_diagnostics()

    def _tick_preflight(self, now: float) -> None:
        """Wait for the preflight node's verdict, then arm or abort."""
        if not self._preflight_ready:
            return
        if self._setpoint_stream_started is None:
            self._setpoint_stream_started = now
            return
        if now - self._setpoint_stream_started < OFFBOARD_WARMUP_S:
            # Keep streaming. Requesting OFFBOARD before PX4 has seen a steady
            # setpoint stream gets the request rejected, and retrying in a tight
            # loop just fills the MAVLink link with denied commands.
            return
        self._sm.request(MissionState.ARMING, "preflight ready", now)
        self._call_set_mode("OFFBOARD")
        self._call_arm(True)

    def _tick_arming(self, now: float) -> None:
        """Advance once the autopilot actually reports armed."""
        if self._state is not None and self._state.armed:
            self._sm.confirm_armed(now)
            self._call_takeoff(self._takeoff_altitude())

    def _tick_takeoff(self, now: float) -> None:
        """Advance once we are within a metre of the takeoff altitude."""
        target = self._takeoff_altitude()
        if self._pose is None:
            return
        if self._pose.pose.position.z >= target - 1.0:
            self._sm.confirm_takeoff_complete(now)
            self._waypoint_reached_at = now

    def _tick_mission(self, now: float) -> None:
        """Fly the expanded waypoint list, one acceptance radius at a time."""
        if not self._waypoints or self._pose is None or self._mission is None:
            return
        index = min(self._sm.waypoint_index, len(self._waypoints) - 1)
        target = self._waypoints[index]
        east, north, up = self._mission.waypoint_to_enu(target)
        # MAVROS local_position/pose is ENU relative to the EKF origin, which we
        # assume matches the mission origin. If it does not, the mission flies
        # offset by exactly that difference -- see docs/TF_TREE.md.
        dx = east - self._pose.pose.position.x
        dy = north - self._pose.pose.position.y
        dz = up - self._pose.pose.position.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance <= target.acceptance_radius:
            if self._waypoint_reached_at is None:
                self._waypoint_reached_at = now
            if now - self._waypoint_reached_at >= target.hold_time:
                self._waypoint_reached_at = None
                if not self._sm.advance_waypoint():
                    self._sm.finish_mission(now)
                    self._call_set_mode("AUTO.RTL")
            return

        self._waypoint_reached_at = None
        timeout = float(self.get_parameter("waypoint_timeout_s").value)
        oldest = self._sm.history[-1].timestamp if self._sm.history else now
        if now - oldest > timeout:
            self._safe_abort(
                AbortReason.OPERATOR,
                f"waypoint {index} not reached in {timeout:.0f} s "
                f"(still {distance:.0f} m away)",
            )

    def _tick_rtl(self, now: float) -> None:
        """Hand RTL to the autopilot and wait for it to start descending."""
        if self._pose is not None and self._pose.pose.position.z < 1.0:
            self._sm.begin_landing(now)
            self._call_land()

    def _tick_landing(self, now: float) -> None:
        """Finish once the autopilot reports disarmed."""
        if self._state is not None and not self._state.armed:
            self._sm.confirm_disarmed(now)
            self.get_logger().info("mission complete, vehicle disarmed")

    # -- outputs ------------------------------------------------------------
    def _takeoff_altitude(self) -> float:
        """Takeoff altitude from the mission, or the parameter fallback."""
        if self._waypoints:
            return self._waypoints[0].altitude
        return float(self.get_parameter("takeoff_altitude_m").value)

    def _publish_setpoint(self, now: float) -> None:
        """Publish a local ENU setpoint every tick, without exception.

        Even in IDLE. PX4 needs a warm stream before it will accept OFFBOARD,
        and it drops OFFBOARD if the stream stops for ~0.5 s.
        """
        if self._setpoint_stream_started is None and self._sm.state is not MissionState.IDLE:
            self._setpoint_stream_started = now

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        if self._waypoints and self._mission is not None:
            index = min(self._sm.waypoint_index, len(self._waypoints) - 1)
            target = self._waypoints[index]
            east, north, up = self._mission.waypoint_to_enu(target)
            msg.pose.position.x = east
            msg.pose.position.y = north
            msg.pose.position.z = up
            if target.yaw_deg is not None:
                # Mission yaw is a compass heading (NED). MAVROS setpoints are
                # ENU, so convert rather than passing the number through.
                yaw_enu = ned_yaw_to_enu_yaw(math.radians(target.yaw_deg))
                qx, qy, qz, qw = quat_to_ros_xyzw(quat_from_euler(0.0, 0.0, yaw_enu))
                msg.pose.orientation.x = qx
                msg.pose.orientation.y = qy
                msg.pose.orientation.z = qz
                msg.pose.orientation.w = qw
            else:
                msg.pose.orientation.w = 1.0
        elif self._pose is not None:
            # No mission: hold where we are. Publishing zeros here would command
            # a dive to the EKF origin the instant OFFBOARD engaged.
            msg.pose = self._pose.pose
        else:
            msg.pose.orientation.w = 1.0

        self._pub_setpoint.publish(msg)

    def _publish_state(self) -> None:
        msg = String()
        snapshot = self._sm.snapshot()
        msg.data = " ".join(f"{k}={v}" for k, v in snapshot.items())
        self._pub_state.publish(msg)

    def _publish_diagnostics(self) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        item = DiagnosticStatus(
            name="drone_bringup: mission_executor", hardware_id="companion"
        )
        item.values = [
            KeyValue(key="state", value=self._sm.state.value),
            KeyValue(
                key="waypoint",
                value=f"{self._sm.waypoint_index}/{self._sm.waypoint_count}",
            ),
            KeyValue(key="preflight_ready", value=str(self._preflight_ready)),
            KeyValue(key="geofence_breach", value=str(self._geofence_breach)),
            KeyValue(key="site_mismatch", value=str(self._site_mismatch)),
            KeyValue(key="has_fix", value=str(self._fix is not None)),
            KeyValue(
                key="abort_reason",
                value="" if self._sm.abort_reason is None else self._sm.abort_reason.value,
            ),
        ]
        if self._sm.state is MissionState.FAULT:
            item.level = DiagnosticStatus.ERROR
        elif self._sm.abort_reason is not None:
            item.level = DiagnosticStatus.WARN
        else:
            item.level = DiagnosticStatus.OK
        item.message = f"state {self._sm.state.value}"
        array.status = [item]
        self._pub_diag.publish(array)

    # -- MAVROS service calls ----------------------------------------------
    def _service_ready(self, client, name: str) -> bool:
        """True if a service is up. Never blocks the executor thread for long.

        ``wait_for_service`` inside a timer callback blocks the single-threaded
        executor, which stops the setpoint stream, which drops OFFBOARD. So this
        uses ``service_is_ready`` and reports rather than waiting.
        """
        if client.service_is_ready():
            return True
        self.get_logger().warn(f"MAVROS service {name} not available")
        return False

    def _call_arm(self, arm: bool) -> None:
        """Async arm/disarm. The result is confirmed via ``/mavros/state``."""
        if not self._service_ready(self._cli_arm, "cmd/arming"):
            return
        request = CommandBool.Request()
        request.value = arm
        future = self._cli_arm.call_async(request)
        future.add_done_callback(
            lambda f: self._on_arm_result(f, arm)
        )

    def _on_arm_result(self, future, arm: bool) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - service errors are opaque
            self.get_logger().error(f"arming service call failed: {exc}")
            return
        if result is not None and not result.success:
            self.get_logger().warn(
                f"autopilot rejected {'arm' if arm else 'disarm'} "
                f"(result={result.result})"
            )
            if self._sm.state is MissionState.ARMING:
                self._sm.arming_rejected(f"result={result.result}", self._now())

    def _call_set_mode(self, mode: str) -> None:
        """Async flight-mode change."""
        if not self._service_ready(self._cli_mode, "set_mode"):
            return
        request = SetMode.Request()
        request.custom_mode = mode
        self._cli_mode.call_async(request)
        self.get_logger().info(f"requested flight mode {mode}")

    def _call_takeoff(self, altitude: float) -> None:
        """Async takeoff command."""
        if not self._service_ready(self._cli_takeoff, "cmd/takeoff"):
            return
        request = CommandTOL.Request()
        request.altitude = float(altitude)
        self._cli_takeoff.call_async(request)
        self.get_logger().info(f"takeoff to {altitude:.1f} m requested")

    def _call_land(self) -> None:
        """Async land command."""
        if not self._service_ready(self._cli_land, "cmd/land"):
            return
        self._cli_land.call_async(CommandTOL.Request())
        self.get_logger().info("land requested")

    # -- shutdown -----------------------------------------------------------
    def destroy_node(self) -> bool:
        """Stop the tick before publishers are torn down."""
        self.deactivate()
        return super().destroy_node()


def main(args=None) -> None:
    """Console-script entry point."""
    rclpy.init(args=args)
    node = MissionExecutorNode()
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
