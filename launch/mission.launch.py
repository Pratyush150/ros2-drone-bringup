"""Load and run one mission against an already-running MAVROS.

Use this when the vehicle (real or SITL) is already up and you just want to
swap the mission plan. It starts only the mission executor and the geofence
monitor, so it will not fight an existing telemetry bridge or preflight node
for topic names.

The mission file is validated at node construction. If it is malformed, the
node logs the exact item and field and then idles -- it will not arm on a
half-parsed plan.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the mission launch description."""
    pkg = FindPackageShare("drone_bringup")

    args = [
        DeclareLaunchArgument(
            "mission_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_mission.yaml"]),
            description="Mission YAML to execute.",
        ),
        DeclareLaunchArgument(
            "geofence_file",
            default_value=PathJoinSubstitution([pkg, "config", "example_geofence.yaml"]),
            description="Geofence YAML the monitor loads.",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="ROS namespace, matching the running bringup.",
        ),
        DeclareLaunchArgument(
            "mavros_namespace",
            default_value="/mavros",
            description="Namespace the running MAVROS publishes under.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="True when running against SITL with Gazebo's /clock.",
        ),
        DeclareLaunchArgument(
            "auto_start",
            default_value="false",
            description="Start as soon as preflight passes. Otherwise publish "
            "'start' on <ns>/mission_executor/command.",
        ),
        DeclareLaunchArgument(
            "start_geofence_monitor",
            default_value="true",
            description="Set false if a geofence monitor is already running.",
        ),
    ]

    namespace = LaunchConfiguration("namespace")
    mavros_ns = LaunchConfiguration("mavros_namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_dir = PathJoinSubstitution([pkg, "config"])

    nodes = GroupAction(
        actions=[
            PushRosNamespace(namespace),
            Node(
                package="drone_bringup",
                executable="geofence_monitor",
                name="geofence_monitor",
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_geofence_monitor")),
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
                executable="mission_executor",
                name="mission_executor",
                output="screen",
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
