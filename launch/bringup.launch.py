"""Core bringup: the four drone_bringup nodes, namespaced for multi-vehicle.

This is the piece that both ``sitl.launch.py`` and ``hardware.launch.py``
include. It does not start MAVROS, PX4, or Gazebo -- those differ between sim
and hardware, and keeping them out of here is what lets one file describe the
application and the other two describe the environment.

Multi-vehicle: everything below sits inside ``PushRosNamespace(namespace)``, so
launching this twice with ``namespace:=uav1`` and ``namespace:=uav2`` gives two
complete, non-colliding node graphs. Note the cross-node topics are declared as
parameters rather than hard-coded absolute names, precisely so the namespace
push does not break the wiring.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the bringup launch description."""
    pkg = FindPackageShare("drone_bringup")

    args = [
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="ROS namespace for every node. Set it per vehicle "
            "(uav1, uav2, ...) for a multi-vehicle setup.",
        ),
        DeclareLaunchArgument(
            "mavros_namespace",
            default_value="/mavros",
            description="Namespace MAVROS publishes under. For multi-vehicle "
            "this is usually /uav1/mavros.",
        ),
        DeclareLaunchArgument(
            "mission_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_mission.yaml"]),
            description="Mission YAML to load into the executor.",
        ),
        DeclareLaunchArgument(
            "geofence_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_geofence.yaml"]),
            description="Geofence YAML to load into the monitor.",
        ),
        DeclareLaunchArgument(
            "params_dir",
            default_value=PathJoinSubstitution([pkg, "config"]),
            description="Directory holding the per-node parameter YAML files.",
        ),
        DeclareLaunchArgument(
            "auto_start",
            default_value="false",
            description="Start the mission automatically once preflight passes. "
            "Leave false on a real vehicle.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use /clock instead of the wall clock. Must be true when "
            "Gazebo is publishing /clock, or every timestamp comparison in the "
            "stack is meaningless.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="ROS log level for every node in this file.",
        ),
    ]

    namespace = LaunchConfiguration("namespace")
    mavros_ns = LaunchConfiguration("mavros_namespace")
    params_dir = LaunchConfiguration("params_dir")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")
    common_args = ["--ros-args", "--log-level", log_level]

    nodes = GroupAction(
        actions=[
            PushRosNamespace(namespace),
            Node(
                package="drone_bringup",
                executable="telemetry_bridge",
                name="telemetry_bridge",
                output="screen",
                arguments=common_args,
                parameters=[
                    PathJoinSubstitution([params_dir, "telemetry_bridge.yaml"]),
                    {
                        "mavros_namespace": mavros_ns,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                package="drone_bringup",
                executable="geofence_monitor",
                name="geofence_monitor",
                output="screen",
                arguments=common_args,
                parameters=[
                    PathJoinSubstitution([params_dir, "geofence_monitor.yaml"]),
                    {
                        "mavros_namespace": mavros_ns,
                        "geofence_file": LaunchConfiguration("geofence_file"),
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                package="drone_bringup",
                executable="preflight_check",
                name="preflight_check",
                output="screen",
                arguments=common_args,
                parameters=[
                    PathJoinSubstitution([params_dir, "preflight_check.yaml"]),
                    {
                        "mavros_namespace": mavros_ns,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                package="drone_bringup",
                executable="mission_executor",
                name="mission_executor",
                output="screen",
                arguments=common_args,
                parameters=[
                    PathJoinSubstitution([params_dir, "mission_executor.yaml"]),
                    {
                        "mavros_namespace": mavros_ns,
                        "mission_file": LaunchConfiguration("mission_file"),
                        "auto_start": LaunchConfiguration("auto_start"),
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
        ]
    )

    return LaunchDescription([*args, nodes])
