"""ament_python setup for the drone_bringup package.

Data files matter here. ``colcon build`` will happily produce a package whose
launch files are not installed, and then ``ros2 launch drone_bringup
sitl.launch.py`` fails with "file not found" while the file is sitting right
there in the source tree. Every directory you want at runtime -- launch, config,
worlds, models -- has to be listed below.
"""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "drone_bringup"


def _model_files():
    """Install every SDF/config file under models/, preserving its subdirectory.

    Gazebo resolves a model by looking for ``model.config`` in a directory named
    after the model, so the directory structure has to survive installation.
    """
    entries = []
    for root, _dirs, files in os.walk("models"):
        if not files:
            continue
        entries.append(
            (
                os.path.join("share", package_name, root),
                [os.path.join(root, f) for f in files],
            )
        )
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/config", glob("config/*.rviz")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        *_model_files(),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Pratyush Vatsa",
    maintainer_email="pratyush@example.invalid",
    description=(
        "ROS 2 bringup for a PX4/ArduPilot multirotor: MAVROS telemetry "
        "normalisation, guarded mission state machine, predictive geofence, "
        "preflight gating, and PX4 SITL launch files."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_executor = drone_bringup.nodes.mission_executor_node:main",
            "telemetry_bridge = drone_bringup.nodes.telemetry_bridge_node:main",
            "geofence_monitor = drone_bringup.nodes.geofence_monitor_node:main",
            "preflight_check = drone_bringup.nodes.preflight_check_node:main",
        ],
    },
)
