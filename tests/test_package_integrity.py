"""Tests that the packaging, launch, config, and simulation assets are coherent.

None of this needs a ROS installation. It catches the class of mistake that
otherwise only shows up as "ros2 launch: file not found" ten minutes into a
field trip: an entry point pointing at a module that was renamed, a launch file
referencing a config that is not installed, a params file whose node name does
not match the executable, or an SDF that stopped being well-formed.
"""

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent

NODE_MODULES = {
    "mission_executor": "drone_bringup/nodes/mission_executor_node.py",
    "telemetry_bridge": "drone_bringup/nodes/telemetry_bridge_node.py",
    "geofence_monitor": "drone_bringup/nodes/geofence_monitor_node.py",
    "preflight_check": "drone_bringup/nodes/preflight_check_node.py",
}


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


class TestPackageXml:
    """package.xml must be format 3 and declare what the code imports."""

    @pytest.fixture
    def root(self):
        return ET.fromstring(read("package.xml"))

    def test_is_format_three(self, root):
        assert root.attrib["format"] == "3"

    def test_name_matches_the_python_package(self, root):
        assert root.findtext("name") == "drone_bringup"
        assert (REPO / "drone_bringup" / "__init__.py").is_file()

    def test_build_type_is_ament_python(self, root):
        assert root.find("export").findtext("build_type") == "ament_python"

    def test_buildtool_depend(self, root):
        assert root.findtext("buildtool_depend") == "ament_python"

    def test_license_is_mit(self, root):
        assert root.findtext("license") == "MIT"

    @pytest.mark.parametrize(
        "dependency",
        [
            "rclpy",
            "mavros_msgs",
            "sensor_msgs",
            "geometry_msgs",
            "nav_msgs",
            "tf2_ros",
            "vision_msgs",
            "std_msgs",
            "diagnostic_msgs",
            "visualization_msgs",
            "rcl_interfaces",
        ],
    )
    def test_required_dependency_is_declared(self, root, dependency):
        declared = {e.text for e in root.iter() if e.tag.endswith("depend")}
        assert dependency in declared

    def test_every_imported_ros_package_is_declared(self, root):
        """Any `from <pkg>.msg import ...` in the nodes must appear in package.xml."""
        declared = {e.text for e in root.iter() if e.tag.endswith("depend")}
        imported = set()
        for module in NODE_MODULES.values():
            tree = ast.parse(read(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    head = node.module.split(".")[0]
                    if head.endswith("_msgs") or head in {"rclpy", "rcl_interfaces"}:
                        imported.add(head)
        missing = imported - declared
        assert not missing, f"imported but not declared in package.xml: {sorted(missing)}"

    def test_resource_marker_exists(self):
        assert (REPO / "resource" / "drone_bringup").is_file()


class TestSetupPy:
    """Entry points and data_files must match what is on disk."""

    @pytest.fixture
    def source(self):
        return read("setup.py")

    @pytest.mark.parametrize("executable,module", sorted(NODE_MODULES.items()))
    def test_console_script_targets_a_real_module(self, source, executable, module):
        dotted = module.replace("/", ".").removesuffix(".py")
        assert f"{executable} = {dotted}:main" in source
        assert (REPO / module).is_file()

    @pytest.mark.parametrize("module", sorted(NODE_MODULES.values()))
    def test_every_node_module_defines_main(self, module):
        tree = ast.parse(read(module))
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "main" in names

    def test_launch_config_worlds_are_installed(self, source):
        for directory in ("launch", "config", "worlds"):
            assert f"share/{{package_name}}/{directory}" in source or (
                f'f"share/{{package_name}}/{directory}"' in source
            )

    def test_resource_index_entry(self, source):
        assert "share/ament_index/resource_index/packages" in source


class TestLaunchFiles:
    """Launch files must parse and expose a generate_launch_description."""

    LAUNCH_FILES = ["bringup", "sitl", "hardware", "mission"]

    @pytest.mark.parametrize("name", LAUNCH_FILES)
    def test_file_exists_and_parses(self, name):
        tree = ast.parse(read(f"launch/{name}.launch.py"))
        functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "generate_launch_description" in functions

    @pytest.mark.parametrize("name", LAUNCH_FILES)
    def test_declares_launch_arguments(self, name):
        source = read(f"launch/{name}.launch.py")
        assert source.count("DeclareLaunchArgument") >= 3

    @pytest.mark.parametrize("name", LAUNCH_FILES)
    def test_every_argument_has_a_description(self, name):
        """A launch argument with no description is invisible in `--show-args`."""
        source = read(f"launch/{name}.launch.py")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DeclareLaunchArgument"
            ):
                keywords = {kw.arg for kw in node.keywords}
                assert "description" in keywords

    @pytest.mark.parametrize("name", ["bringup", "sitl", "hardware", "mission"])
    def test_supports_a_namespace_argument(self, name):
        assert '"namespace"' in read(f"launch/{name}.launch.py")

    def test_sitl_uses_the_offboard_port_not_the_gcs_port(self):
        """MAVROS binds 14540 (offboard). 14550 belongs to QGroundControl.

        Checked against the AST with the module docstring stripped, so the
        explanation of *why* 14550 is off limits does not itself trip the test.
        """
        tree = ast.parse(read("launch/sitl.launch.py"))
        if tree.body and isinstance(tree.body[0], ast.Expr):
            tree.body = tree.body[1:]  # drop the module docstring
        code = ast.unparse(tree)
        assert "14540" in code
        # A URL literal is the only place a port becomes a binding. Prose that
        # explains why 14550 is off limits is fine and should stay.
        urls = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "udp" in node.value
        ]
        assert not any("14550" in url for url in urls), urls

    def test_hardware_launch_does_not_default_to_ttyacm(self):
        source = read("launch/hardware.launch.py")
        match = re.search(r'default_value="(/dev/[^"]+)"', source)
        assert match is not None
        assert "ttyACM" not in match.group(1)


class TestConfigFiles:
    """Every params file must parse and target the node it is named for."""

    PARAM_FILES = {
        "telemetry_bridge.yaml": "telemetry_bridge",
        "preflight_check.yaml": "preflight_check",
        "geofence_monitor.yaml": "geofence_monitor",
        "mission_executor.yaml": "mission_executor",
    }

    @pytest.mark.parametrize("filename,node", sorted(PARAM_FILES.items()))
    def test_params_file_parses_and_names_the_node(self, filename, node):
        data = yaml.safe_load(read(f"config/{filename}"))
        assert list(data) == [f"/**/{node}"]
        assert "ros__parameters" in data[f"/**/{node}"]

    @pytest.mark.parametrize("filename,node", sorted(PARAM_FILES.items()))
    def test_declared_parameters_cover_the_params_file(self, filename, node):
        """Every key in the YAML must be a parameter the node actually declares.

        A typo'd key in a params file is silently ignored by ROS 2 unless the
        node is set to reject undeclared parameters, so this is the only place
        it gets caught.
        """
        module = NODE_MODULES[node]
        source = read(module)
        declared = set(re.findall(r'declare_parameter\(\s*\n?\s*"([a-z0-9_]+)"', source))
        data = yaml.safe_load(read(f"config/{filename}"))
        keys = set(data[f"/**/{node}"]["ros__parameters"])
        assert keys <= declared, f"{filename}: undeclared parameters {sorted(keys - declared)}"

    def test_params_use_the_wildcard_namespace(self):
        """`/**/node` survives being pushed into a per-vehicle namespace."""
        for filename in self.PARAM_FILES:
            assert read(f"config/{filename}").count("/**/") == 1

    def test_rviz_config_parses(self):
        data = yaml.safe_load(read("config/drone_bringup.rviz"))
        assert data["Visualization Manager"]["Global Options"]["Fixed Frame"] == "map"

    def test_rviz_sensor_topics_use_best_effort(self):
        """RViz defaults to RELIABLE; a BEST_EFFORT publisher would never match."""
        data = yaml.safe_load(read("config/drone_bringup.rviz"))
        displays = data["Visualization Manager"]["Displays"]
        odom = next(d for d in displays if d.get("Name") == "Odometry")
        assert odom["Topic"]["Reliability Policy"] == "Best Effort"

    def test_rviz_latched_topics_use_transient_local(self):
        data = yaml.safe_load(read("config/drone_bringup.rviz"))
        displays = data["Visualization Manager"]["Displays"]
        for name in ("Geofence", "MissionPath"):
            display = next(d for d in displays if d.get("Name") == name)
            assert display["Topic"]["Durability Policy"] == "Transient Local"


class TestSimulationAssets:
    """The world and model files must be well-formed and internally consistent."""

    def test_world_parses(self):
        root = ET.fromstring(read("worlds/survey_field.sdf"))
        assert root.tag == "sdf"
        assert root.find("world").attrib["name"] == "survey_field"

    def test_world_origin_matches_the_example_mission(self):
        root = ET.fromstring(read("worlds/survey_field.sdf"))
        coords = root.find("world").find("spherical_coordinates")
        mission = yaml.safe_load(read("config/example_mission.yaml"))
        assert float(coords.findtext("latitude_deg")) == pytest.approx(
            mission["origin"]["latitude"]
        )
        assert float(coords.findtext("longitude_deg")) == pytest.approx(
            mission["origin"]["longitude"]
        )
        assert float(coords.findtext("elevation")) == pytest.approx(
            mission["origin"]["altitude"]
        )

    def test_world_uses_enu_orientation(self):
        root = ET.fromstring(read("worlds/survey_field.sdf"))
        coords = root.find("world").find("spherical_coordinates")
        assert coords.findtext("world_frame_orientation") == "ENU"

    def test_world_includes_the_landing_pad(self):
        source = read("worlds/survey_field.sdf")
        assert "model://landing_pad" in source

    def test_model_config_points_at_the_sdf(self):
        root = ET.fromstring(read("models/landing_pad/model.config"))
        assert root.findtext("name") == "landing_pad"
        assert root.findtext("sdf") == "model.sdf"
        assert (REPO / "models" / "landing_pad" / "model.sdf").is_file()

    def test_landing_pad_parses_and_is_static(self):
        root = ET.fromstring(read("models/landing_pad/model.sdf"))
        model = root.find("model")
        assert model.attrib["name"] == "landing_pad"
        assert model.findtext("static") == "true"

    def test_landing_pad_has_a_collision_and_visuals(self):
        root = ET.fromstring(read("models/landing_pad/model.sdf"))
        link = root.find("model").find("link")
        assert link.find("collision") is not None
        assert len(link.findall("visual")) >= 4

    def test_no_binary_assets_are_committed(self):
        """Text-only repo: catches a stray mesh, texture, or bag file."""
        binary_suffixes = {".dae", ".stl", ".png", ".jpg", ".jpeg", ".bin", ".db3", ".pyc"}
        offenders = [
            str(p.relative_to(REPO))
            for p in REPO.rglob("*")
            if p.is_file()
            and p.suffix.lower() in binary_suffixes
            and "__pycache__" not in p.parts
        ]
        assert not offenders, f"binary assets found: {offenders}"


class TestCoreHasNoRosImports:
    """The load-bearing rule: drone_bringup.core must never import ROS."""

    CORE_MODULES = [
        "drone_bringup/core/__init__.py",
        "drone_bringup/core/geodesy.py",
        "drone_bringup/core/frames.py",
        "drone_bringup/core/mission.py",
        "drone_bringup/core/geofence.py",
        "drone_bringup/core/state_machine.py",
    ]

    FORBIDDEN_PREFIXES = ("rclpy", "rcl_interfaces", "tf2", "ament", "launch")

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_no_ros_imports(self, module):
        tree = ast.parse(read(module))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                head = name.split(".")[0]
                assert not head.endswith("_msgs"), f"{module} imports {name}"
                assert not head.startswith(self.FORBIDDEN_PREFIXES), (
                    f"{module} imports {name}"
                )

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_module_has_a_docstring(self, module):
        assert ast.get_docstring(ast.parse(read(module)))

    @pytest.mark.parametrize("module", CORE_MODULES)
    def test_every_public_function_has_a_docstring(self, module):
        tree = ast.parse(read(module))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                assert ast.get_docstring(node), f"{module}:{node.name} has no docstring"
            if isinstance(node, ast.ClassDef):
                assert ast.get_docstring(node), f"{module}:{node.name} has no docstring"
