"""Bringup against a real flight controller over a serial or UDP link.

Differences from SITL that actually matter
------------------------------------------
* **The serial device name is not stable.** ``/dev/ttyACM0`` becomes
  ``/dev/ttyACM1`` after a replug, a brownout, or a USB hub renumbering, and
  your bringup then talks to nothing. Use a udev-created symlink. There is a
  rule in the docstring below; put it in
  ``/etc/udev/rules.d/99-px4.rules`` and point ``fcu_url`` at ``/dev/px4fmu``.
* **Baud rate must match the autopilot's TELEM/USB setting.** 921600 is the
  usual companion-computer rate over a TELEM port; USB CDC ACM ignores baud but
  MAVROS still wants the field.
* **use_sim_time must be false.** If it is true and nothing publishes /clock,
  every ROS timestamp stays at zero, every staleness check reads "0 seconds
  old", and the whole telemetry-freshness layer silently stops working.
* **auto_start defaults to false.** A vehicle that arms itself when a launch
  file starts is a vehicle that arms itself when you plug in a battery to debug
  something else.

Suggested udev rule (adjust idVendor/idProduct from ``lsusb``)::

    SUBSYSTEM=="tty", ATTRS{idVendor}=="1209", ATTRS{idProduct}=="5741", \
      SYMLINK+="px4fmu", MODE="0666"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the hardware launch description."""
    pkg = FindPackageShare("drone_bringup")

    args = [
        DeclareLaunchArgument(
            "fcu_url",
            default_value="/dev/px4fmu:921600",
            description="MAVROS FCU URL. Serial: /dev/px4fmu:921600. UDP over a "
            "telemetry radio bridge: udp://:14540@. Use a udev symlink, not "
            "/dev/ttyACM0 -- that name moves.",
        ),
        DeclareLaunchArgument(
            "gcs_url",
            default_value="udp://@",
            description="Where MAVROS forwards a GCS stream. 'udp://@' broadcasts "
            "to 14550 on the local network so QGroundControl finds the vehicle. "
            "Set it to '' to forward nothing.",
        ),
        DeclareLaunchArgument(
            "tgt_system",
            default_value="1",
            description="MAVLink system id of the autopilot.",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="ROS namespace for every node.",
        ),
        DeclareLaunchArgument(
            "start_mavros",
            default_value="true",
            description="Start MAVROS from here.",
        ),
        DeclareLaunchArgument(
            "mission_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_mission.yaml"]),
            description="Mission YAML to load. Replace this with your own site's "
            "mission; the example is over a field in Switzerland.",
        ),
        DeclareLaunchArgument(
            "geofence_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_geofence.yaml"]),
            description="Geofence YAML to load.",
        ),
        DeclareLaunchArgument(
            "auto_start",
            default_value="false",
            description="Begin the mission as soon as preflight passes. Keep this "
            "false unless you have a specific reason and a safety pilot.",
        ),
    ]

    namespace = LaunchConfiguration("namespace")

    mavros = Node(
        package="mavros",
        executable="mavros_node",
        name="mavros",
        namespace=[namespace, "/mavros"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_mavros")),
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            {
                "fcu_url": LaunchConfiguration("fcu_url"),
                "gcs_url": LaunchConfiguration("gcs_url"),
                "target_system_id": LaunchConfiguration("tgt_system"),
                "target_component_id": 1,
                "system_id": 1,
                "component_id": 191,
                "use_sim_time": False,
                # Ask PX4 for the rates we need instead of accepting defaults.
                # A 1 Hz local position stream makes a 20 Hz offboard loop
                # extrapolate, and extrapolating position is how you get a
                # vehicle that oscillates and blames the controller.
                "conn/timeout": 10.0,
                "conn/heartbeat_rate": 1.0,
            }
        ],
    )

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, "launch", "bringup.launch.py"])
        ),
        launch_arguments={
            "namespace": namespace,
            "mavros_namespace": [namespace, "/mavros"],
            "mission_file": LaunchConfiguration("mission_file"),
            "geofence_file": LaunchConfiguration("geofence_file"),
            "use_sim_time": "false",
            "auto_start": LaunchConfiguration("auto_start"),
        }.items(),
    )

    return LaunchDescription(
        [
            *args,
            LogInfo(
                msg=[
                    "hardware bringup: fcu_url=",
                    LaunchConfiguration("fcu_url"),
                    " -- if this device name is /dev/ttyACM0, expect it to move.",
                ]
            ),
            mavros,
            bringup,
        ]
    )
