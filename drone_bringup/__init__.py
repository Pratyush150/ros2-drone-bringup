"""ROS 2 bringup package for a PX4/ArduPilot multirotor.

The :mod:`drone_bringup.core` subpackage holds every piece of logic that does
not need ROS -- geodesy, frame conversions, mission parsing, geofencing, and
the executor state machine -- so it can be unit tested with plain ``pytest`` on
a machine with no ROS installation. :mod:`drone_bringup.nodes` holds the rclpy
nodes that wire that logic to MAVROS topics.
"""

__version__ = "0.1.0"
