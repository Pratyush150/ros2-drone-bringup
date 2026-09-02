"""PX4 SITL + Gazebo + MAVROS + this package's bringup, in one command.

What this actually starts
-------------------------
1. **PX4 SITL with Gazebo.** Launched via ``PX4_DIR/build/px4_sitl_default/bin/px4``
   through the ``make px4_sitl gz_x500`` style entry point. PX4 owns the Gazebo
   process, which is why there is no separate ``gz sim`` action here -- starting
   your own Gazebo *and* letting PX4 start one gives you two simulators fighting
   over the same world, and the symptom is a vehicle that will not arm because
   its sensors never update.
2. **MAVROS**, pointed at UDP 14540.
3. **The bringup nodes** from ``bringup.launch.py``.

Ports -- read docs/SIMULATION.md before changing these
------------------------------------------------------
PX4 SITL opens two separate MAVLink streams:

* **udp://:14540** -- the *offboard* / onboard API port. MAVSDK, MAVROS, and
  anything doing autonomy connects here.
* **udp://:14550** -- the *GCS* port. QGroundControl broadcasts to and listens
  on this one.

They are separate on purpose. Point MAVROS at 14550 and it fights QGC for the
same socket: on Linux the second binder usually gets the packets and the first
silently goes quiet, so QGC shows "waiting for vehicle" while MAVROS looks fine,
or the reverse. If you need a third consumer, add a mavlink-router instance --
do not double-bind a port.

For multi-vehicle, PX4 offsets both ports by the instance index: vehicle N uses
14540+N and 14550+N. The ``instance`` argument below applies that offset.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the SITL launch description."""
    pkg = FindPackageShare("drone_bringup")

    args = [
        DeclareLaunchArgument(
            "px4_dir",
            default_value=os.environ.get("PX4_DIR", os.path.expanduser("~/PX4-Autopilot")),
            description="Path to a built PX4-Autopilot checkout. Override with "
            "the PX4_DIR environment variable or this argument.",
        ),
        DeclareLaunchArgument(
            "px4_model",
            default_value="gz_x500",
            description="PX4 SITL airframe target, e.g. gz_x500, gz_x500_depth, "
            "gz_rc_cessna. Must match a target your PX4 checkout can build.",
        ),
        DeclareLaunchArgument(
            "world",
            default_value="survey_field",
            description="Gazebo world name. 'survey_field' is the one shipped in "
            "this package's worlds/ directory; add it to GZ_SIM_RESOURCE_PATH.",
        ),
        DeclareLaunchArgument(
            "instance",
            default_value="0",
            description="PX4 SITL instance index. Offsets the MAVLink ports: "
            "offboard 14540+N, GCS 14550+N.",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="ROS namespace for the bringup nodes.",
        ),
        DeclareLaunchArgument(
            "start_px4",
            default_value="true",
            description="Start PX4 SITL from here. Set false if you already have "
            "'make px4_sitl gz_x500' running in another terminal -- which is the "
            "saner workflow while you are iterating on the sim.",
        ),
        DeclareLaunchArgument(
            "start_mavros",
            default_value="true",
            description="Start MAVROS. Set false if you run it yourself.",
        ),
        DeclareLaunchArgument(
            "mission_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_mission.yaml"]),
            description="Mission YAML to load.",
        ),
        DeclareLaunchArgument(
            "geofence_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_geofence.yaml"]),
            description="Geofence YAML to load.",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Start RViz with this package's config.",
        ),
    ]

    px4_dir = LaunchConfiguration("px4_dir")
    instance = LaunchConfiguration("instance")
    namespace = LaunchConfiguration("namespace")

    # Offboard port = 14540 + instance; GCS port = 14550 + instance.
    offboard_port = PythonExpression(["str(14540 + int('", instance, "'))"])

    px4 = ExecuteProcess(
        cmd=[
            "make",
            "-C",
            px4_dir,
            "px4_sitl",
            LaunchConfiguration("px4_model"),
        ],
        additional_env={
            "PX4_GZ_WORLD": LaunchConfiguration("world"),
            "PX4_SYS_AUTOSTART": "4001",
            "PX4_INSTANCE": instance,
        },
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_px4")),
    )

    mavros = Node(
        package="mavros",
        executable="mavros_node",
        name="mavros",
        namespace=[namespace, "/mavros"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_mavros")),
        parameters=[
            {
                # udp://<bind>@<remote>: bind locally on the offboard port and
                # send back to whoever we heard from. This is the offboard port,
                # NOT the GCS port -- see the module docstring.
                "fcu_url": ["udp://:", offboard_port, "@127.0.0.1:14557"],
                "gcs_url": "",
                "target_system_id": 1,
                "target_component_id": 1,
                # System ID 255 is the conventional GCS id; use 1 for an onboard
                # computer so PX4 does not treat us as a second ground station.
                "system_id": 1,
                "component_id": 191,
                "use_sim_time": True,
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
            "use_sim_time": "true",
            "auto_start": "false",
        }.items(),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", PathJoinSubstitution([pkg, "config", "drone_bringup.rviz"])],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            *args,
            LogInfo(
                msg=[
                    "PX4 SITL offboard port udp://:",
                    offboard_port,
                    " -- leave 14550 free for QGroundControl.",
                ]
            ),
            px4,
            mavros,
            bringup,
            rviz,
        ]
    )
