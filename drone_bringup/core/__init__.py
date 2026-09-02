"""ROS-free logic for the drone bringup package.

Nothing in here imports ``rclpy`` or any ROS message package. That is a hard
rule, not a preference: it is what lets the geometry, the mission format, the
geofence, and the state machine be tested in CI without a ROS distribution, and
it keeps the flight-critical logic reviewable in isolation.
"""

from .frames import (
    flu_to_frd,
    frd_to_flu,
    px4_attitude_to_ros,
    quat_from_euler,
    quat_to_euler,
    ros_attitude_to_px4,
)
from .geodesy import LocalOrigin, enu_to_ned, haversine_distance, ned_to_enu
from .geofence import Geofence, GeofenceStatus, GeofenceZone, ZoneKind
from .mission import Mission, MissionValidationError, Waypoint, generate_lawnmower
from .state_machine import (
    AbortReason,
    MissionState,
    MissionStateMachine,
    PreflightLimits,
    VehicleSnapshot,
    check_preflight,
)

__all__ = [
    "LocalOrigin",
    "enu_to_ned",
    "ned_to_enu",
    "haversine_distance",
    "quat_from_euler",
    "quat_to_euler",
    "frd_to_flu",
    "flu_to_frd",
    "px4_attitude_to_ros",
    "ros_attitude_to_px4",
    "Mission",
    "MissionValidationError",
    "Waypoint",
    "generate_lawnmower",
    "Geofence",
    "GeofenceZone",
    "GeofenceStatus",
    "ZoneKind",
    "MissionState",
    "MissionStateMachine",
    "AbortReason",
    "PreflightLimits",
    "VehicleSnapshot",
    "check_preflight",
]
