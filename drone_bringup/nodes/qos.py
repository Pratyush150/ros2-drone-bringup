"""Shared QoS profiles, and the one ROS 2 gotcha that eats the most time.

Every node in this package imports its QoS from here so the whole graph agrees.
"""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

__all__ = ["sensor_qos", "state_qos", "command_qos"]


# --- QoS ---------------------------------------------------------------------
#
# THE #1 REASON A ROS 2 TOPIC "SILENTLY SHOWS NOTHING"
# ----------------------------------------------------
# In ROS 1, a publisher and a subscriber connected as long as the topic name and
# type matched. In ROS 2 they must also have *compatible QoS*. If they do not,
# DDS simply never forms the match. There is no error, no warning on the default
# log level, and `ros2 topic list` still shows the topic because both endpoints
# exist. `ros2 topic echo` on a BEST_EFFORT publisher prints nothing, because
# echo defaults to RELIABLE. That is the bug people spend an afternoon on.
#
# The compatibility rule is "the subscriber may not demand more than the
# publisher offers":
#   * publisher BEST_EFFORT + subscriber RELIABLE  -> NO MATCH (the classic)
#   * publisher RELIABLE     + subscriber BEST_EFFORT -> matches
#   * publisher VOLATILE     + subscriber TRANSIENT_LOCAL -> NO MATCH
#   * publisher TRANSIENT_LOCAL + subscriber VOLATILE -> matches
#
# Diagnosing it: `ros2 topic info /the/topic --verbose` prints the QoS of every
# endpoint. Compare them. If reliability or durability differ in the offending
# direction, that is your answer.
#
# Why sensor data is BEST_EFFORT: IMU at 200 Hz, GPS at 5 Hz, and pose at 50 Hz
# are streams where the newest sample supersedes the previous one. Retransmitting
# a dropped IMU sample is worse than useless -- by the time it arrives it is
# stale, and the retransmit queue adds latency to the samples you actually want.
# So sensors are BEST_EFFORT + KEEP_LAST(depth 1..10), which is exactly what
# MAVROS publishes with. Subscribe to them with anything RELIABLE and you get
# silence.
#
# Why state/latched data is RELIABLE + TRANSIENT_LOCAL: arming state, mode, and
# the loaded geofence are events, not streams. Missing one leaves you with a
# wrong belief until the next change, which may be never. TRANSIENT_LOCAL also
# means a node that starts late still receives the last value instead of waiting
# for the next transition.


def sensor_qos(depth: int = 5) -> QoSProfile:
    """QoS for high-rate sensor streams. Matches MAVROS and ``rclcpp::SensorDataQoS``."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def state_qos(depth: int = 1) -> QoSProfile:
    """QoS for low-rate latched state: reliable and available to late joiners."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def command_qos(depth: int = 10) -> QoSProfile:
    """QoS for commands and setpoints: reliable, but not replayed to late joiners.

    Replaying an old setpoint to a node that just started is actively dangerous,
    so durability stays VOLATILE here even though reliability is RELIABLE.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )
