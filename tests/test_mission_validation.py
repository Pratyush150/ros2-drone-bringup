"""Tests for mission YAML parsing and validation.

Every failure case asserts on the *message*, not just the exception type. A
validation error that does not tell you which item and which field is wrong is
barely better than no validation at all, so the message content is part of the
contract.
"""

import pytest

from drone_bringup.core.mission import (
    ItemType,
    Mission,
    MissionValidationError,
    load_mission_yaml,
)

MINIMAL = """
name: minimal
origin: {latitude: 47.397742, longitude: 8.545594, altitude: 488.0}
items:
  - {type: takeoff, altitude: 10.0}
  - {type: land}
"""


def _mission(items_yaml: str, header: str = "") -> Mission:
    """Build a mission from an items block plus an optional extra header line."""
    return load_mission_yaml(
        "name: t\n"
        "origin: {latitude: 47.4, longitude: 8.5, altitude: 400.0}\n"
        f"{header}"
        "items:\n" + items_yaml
    )


class TestHappyPath:
    """A well-formed mission parses and expands."""

    def test_minimal_mission_parses(self):
        mission = load_mission_yaml(MINIMAL)
        assert mission.name == "minimal"
        assert mission.item_types == [ItemType.TAKEOFF, ItemType.LAND]

    def test_origin_is_captured(self):
        mission = load_mission_yaml(MINIMAL)
        assert mission.origin.latitude_deg == pytest.approx(47.397742)
        assert mission.origin.altitude_m == pytest.approx(488.0)

    def test_defaults_are_applied(self):
        mission = load_mission_yaml(MINIMAL)
        assert mission.default_speed == pytest.approx(5.0)
        assert mission.default_acceptance_radius == pytest.approx(2.0)

    def test_waypoint_inherits_default_speed(self):
        mission = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: waypoint, latitude: 47.401, longitude: 8.501, altitude: 20.0}\n"
            "  - {type: land}\n",
            header="default_speed: 7.5\n",
        )
        waypoints = mission.expand()
        assert waypoints[1].speed == pytest.approx(7.5)

    def test_waypoint_speed_overrides_default(self):
        mission = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: waypoint, latitude: 47.401, longitude: 8.501, altitude: 20.0,"
            " speed: 3.0}\n"
            "  - {type: land}\n",
            header="default_speed: 7.5\n",
        )
        assert mission.expand()[1].speed == pytest.approx(3.0)

    def test_hold_time_is_carried_through(self):
        mission = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: waypoint, latitude: 47.401, longitude: 8.501, altitude: 20.0,"
            " hold_time: 4.5}\n"
            "  - {type: land}\n"
        )
        assert mission.expand()[1].hold_time == pytest.approx(4.5)

    def test_rtl_expands_to_no_waypoints(self):
        mission = _mission(
            "  - {type: takeoff, altitude: 10.0}\n" "  - {type: rtl}\n"
        )
        assert len(mission.expand()) == 1  # takeoff only

    def test_land_in_place_expands_to_no_waypoint(self):
        mission = load_mission_yaml(MINIMAL)
        assert len(mission.expand()) == 1

    def test_positioned_land_expands_to_a_waypoint(self):
        mission = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: land, latitude: 47.402, longitude: 8.502}\n"
        )
        waypoints = mission.expand()
        assert len(waypoints) == 2
        assert waypoints[-1].altitude == pytest.approx(0.0)

    def test_waypoint_to_enu_uses_relative_altitude(self):
        # The waypoint altitude is metres above the origin. Projecting it must
        # give back exactly that, not (altitude - origin ellipsoid height).
        mission = _mission(
            "  - {type: takeoff, altitude: 42.0}\n" "  - {type: land}\n"
        )
        _, _, up = mission.waypoint_to_enu(mission.expand()[0])
        assert up == pytest.approx(42.0)

    def test_takeoff_is_placed_at_the_origin(self):
        mission = load_mission_yaml(MINIMAL)
        takeoff = mission.expand()[0]
        assert takeoff.latitude == pytest.approx(mission.origin.latitude_deg)
        assert takeoff.longitude == pytest.approx(mission.origin.longitude_deg)

    def test_to_dict_round_trips_the_header(self):
        mission = load_mission_yaml(MINIMAL)
        data = mission.to_dict()
        assert data["name"] == "minimal"
        assert data["origin"]["latitude"] == pytest.approx(47.397742)
        assert len(data["waypoints"]) == len(mission.expand())


class TestStructuralErrors:
    """Root-level and shape problems."""

    def test_root_must_be_a_mapping(self):
        with pytest.raises(MissionValidationError, match="root must be a mapping"):
            load_mission_yaml("- just\n- a\n- list\n")

    def test_missing_name(self):
        with pytest.raises(MissionValidationError, match="'name' must be a non-empty"):
            load_mission_yaml(
                "origin: {latitude: 0, longitude: 0}\nitems: [{type: takeoff, altitude: 5}]\n"
            )

    def test_empty_name(self):
        with pytest.raises(MissionValidationError, match="'name'"):
            load_mission_yaml(
                "name: '   '\norigin: {latitude: 0, longitude: 0}\n"
                "items: [{type: takeoff, altitude: 5}]\n"
            )

    def test_missing_origin(self):
        with pytest.raises(MissionValidationError, match="'origin' must be a mapping"):
            load_mission_yaml("name: x\nitems: [{type: takeoff, altitude: 5}]\n")

    def test_origin_missing_longitude(self):
        with pytest.raises(MissionValidationError, match="missing key 'longitude'"):
            load_mission_yaml(
                "name: x\norigin: {latitude: 47.4}\n"
                "items: [{type: takeoff, altitude: 5}]\n"
            )

    def test_origin_latitude_out_of_range(self):
        with pytest.raises(MissionValidationError, match="mission.origin"):
            load_mission_yaml(
                "name: x\norigin: {latitude: 120.0, longitude: 8.5}\n"
                "items: [{type: takeoff, altitude: 5}]\n"
            )

    def test_items_must_exist(self):
        with pytest.raises(MissionValidationError, match="non-empty list"):
            load_mission_yaml("name: x\norigin: {latitude: 47.4, longitude: 8.5}\n")

    def test_items_must_not_be_empty(self):
        with pytest.raises(MissionValidationError, match="non-empty list"):
            load_mission_yaml(
                "name: x\norigin: {latitude: 47.4, longitude: 8.5}\nitems: []\n"
            )

    def test_item_must_be_a_mapping(self):
        with pytest.raises(MissionValidationError, match=r"items\[0\]: must be a mapping"):
            load_mission_yaml(
                "name: x\norigin: {latitude: 47.4, longitude: 8.5}\nitems: ['takeoff']\n"
            )

    def test_unparseable_yaml(self):
        with pytest.raises(MissionValidationError, match="not parseable"):
            load_mission_yaml("name: [unclosed\n")


class TestItemErrors:
    """Per-item field validation. Each message must locate the problem."""

    def test_missing_type(self):
        pattern = r"items\[0\]: missing required key 'type'"
        with pytest.raises(MissionValidationError, match=pattern):
            _mission("  - {altitude: 10.0}\n")

    def test_unknown_type_lists_the_valid_ones(self):
        with pytest.raises(MissionValidationError) as exc:
            _mission("  - {type: barrel_roll, altitude: 10.0}\n")
        message = str(exc.value)
        assert "barrel_roll" in message
        assert "takeoff" in message and "survey" in message

    def test_takeoff_requires_altitude(self):
        pattern = r"\(takeoff\): missing required key 'altitude'"
        with pytest.raises(MissionValidationError, match=pattern):
            _mission("  - {type: takeoff}\n")

    def test_takeoff_altitude_must_be_positive(self):
        with pytest.raises(MissionValidationError, match=r"takeoff\).altitude: must be > 0"):
            _mission("  - {type: takeoff, altitude: 0.0}\n")

    def test_takeoff_altitude_must_be_a_number(self):
        with pytest.raises(MissionValidationError, match="expected a number, got str"):
            _mission("  - {type: takeoff, altitude: high}\n")

    def test_waypoint_requires_latitude(self):
        pattern = r"items\[1\] \(waypoint\): missing required key 'latitude'"
        with pytest.raises(MissionValidationError, match=pattern):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: waypoint, longitude: 8.5, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_waypoint_latitude_range(self):
        with pytest.raises(MissionValidationError, match=r"latitude must be in \[-90, 90\]"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: waypoint, latitude: 95.0, longitude: 8.5, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_waypoint_longitude_range(self):
        with pytest.raises(MissionValidationError, match=r"longitude must be in \[-180, 180\]"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: waypoint, latitude: 47.4, longitude: 200.0, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_negative_hold_time_is_rejected(self):
        with pytest.raises(MissionValidationError, match=r"hold_time: must be >= 0"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: waypoint, latitude: 47.4, longitude: 8.5, altitude: 20.0,"
                " hold_time: -1.0}\n"
                "  - {type: land}\n"
            )

    def test_zero_speed_is_rejected(self):
        with pytest.raises(MissionValidationError, match=r"speed: must be > 0"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: waypoint, latitude: 47.4, longitude: 8.5, altitude: 20.0,"
                " speed: 0}\n"
                "  - {type: land}\n"
            )

    def test_nan_is_rejected(self):
        with pytest.raises(MissionValidationError, match="must be finite"):
            _mission("  - {type: takeoff, altitude: .nan}\n")

    def test_boolean_is_not_a_number(self):
        # YAML happily parses `true`, and Python happily treats it as 1. A
        # boolean altitude is a typo, not a value.
        with pytest.raises(MissionValidationError, match="expected a number, got bool"):
            _mission("  - {type: takeoff, altitude: true}\n")

    def test_survey_needs_three_vertices(self):
        with pytest.raises(MissionValidationError, match=r"polygon: must be a list of at least 3"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - type: survey\n"
                "    altitude: 30.0\n"
                "    spacing: 20.0\n"
                "    polygon: [[47.4, 8.5], [47.41, 8.5]]\n"
                "  - {type: land}\n"
            )

    def test_survey_vertex_must_be_a_pair(self):
        pattern = r"polygon\[1\]: must be a \[lat, lon\] pair"
        with pytest.raises(MissionValidationError, match=pattern):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - type: survey\n"
                "    altitude: 30.0\n"
                "    spacing: 20.0\n"
                "    polygon: [[47.4, 8.5], [47.41], [47.41, 8.51]]\n"
                "  - {type: land}\n"
            )

    def test_survey_spacing_must_be_positive(self):
        with pytest.raises(MissionValidationError, match=r"spacing: must be > 0"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - type: survey\n"
                "    altitude: 30.0\n"
                "    spacing: -5.0\n"
                "    polygon: [[47.4, 8.5], [47.41, 8.5], [47.41, 8.51]]\n"
                "  - {type: land}\n"
            )

    def test_orbit_requires_radius(self):
        pattern = r"\(orbit\): missing required key 'radius'"
        with pytest.raises(MissionValidationError, match=pattern):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: orbit, latitude: 47.4, longitude: 8.5, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_orbit_needs_at_least_three_points_per_turn(self):
        with pytest.raises(MissionValidationError, match="at least 3 to approximate a circle"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: orbit, latitude: 47.4, longitude: 8.5, altitude: 20.0,"
                " radius: 10.0, points_per_turn: 2}\n"
                "  - {type: land}\n"
            )

    def test_land_needs_both_or_neither_coordinate(self):
        with pytest.raises(MissionValidationError, match="give both 'latitude' and 'longitude'"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: land, latitude: 47.4}\n"
            )


class TestSequenceRules:
    """Whole-mission ordering rules."""

    def test_must_start_with_takeoff(self):
        with pytest.raises(MissionValidationError, match="first item must be 'takeoff'"):
            _mission(
                "  - {type: waypoint, latitude: 47.4, longitude: 8.5, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_must_end_with_land_or_rtl(self):
        with pytest.raises(MissionValidationError, match="last item must be 'land' or 'rtl'"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: waypoint, latitude: 47.4, longitude: 8.5, altitude: 20.0}\n"
            )

    def test_terminal_item_in_the_middle_is_rejected(self):
        with pytest.raises(MissionValidationError, match="terminal item appears before the end"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: rtl}\n"
                "  - {type: waypoint, latitude: 47.4, longitude: 8.5, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_two_takeoffs_are_rejected(self):
        with pytest.raises(MissionValidationError, match="multiple 'takeoff' items"):
            _mission(
                "  - {type: takeoff, altitude: 10.0}\n"
                "  - {type: takeoff, altitude: 20.0}\n"
                "  - {type: land}\n"
            )

    def test_rtl_ending_is_accepted(self):
        mission = _mission("  - {type: takeoff, altitude: 10.0}\n  - {type: rtl}\n")
        assert mission.item_types[-1] is ItemType.RTL


class TestEstimates:
    """Distance and duration estimates."""

    def test_distance_of_a_single_leg(self):
        mission = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: waypoint, latitude: 47.4009, longitude: 8.5, altitude: 10.0}\n"
            "  - {type: land}\n"
        )
        # 0.0009 deg of latitude at 47.4 deg is ~100 m.
        assert mission.total_ground_distance() == pytest.approx(100.0, abs=1.0)

    def test_duration_scales_with_speed(self):
        template = (
            "  - {{type: takeoff, altitude: 10.0}}\n"
            "  - {{type: waypoint, latitude: 47.4009, longitude: 8.5, altitude: 10.0,"
            " speed: {speed}}}\n"
            "  - {{type: land}}\n"
        )
        slow = _mission(template.format(speed=2.0)).estimated_duration_s()
        fast = _mission(template.format(speed=8.0)).estimated_duration_s()
        assert slow == pytest.approx(4.0 * fast, rel=1e-6)

    def test_hold_time_adds_to_duration(self):
        base = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: waypoint, latitude: 47.4009, longitude: 8.5, altitude: 10.0,"
            " speed: 5.0}\n"
            "  - {type: land}\n"
        ).estimated_duration_s()
        held = _mission(
            "  - {type: takeoff, altitude: 10.0}\n"
            "  - {type: waypoint, latitude: 47.4009, longitude: 8.5, altitude: 10.0,"
            " speed: 5.0, hold_time: 30.0}\n"
            "  - {type: land}\n"
        ).estimated_duration_s()
        assert held - base == pytest.approx(30.0, abs=1e-6)


class TestShippedExamples:
    """The example files in config/ must actually be valid."""

    def test_example_mission_loads(self, config_dir):
        from drone_bringup.core.mission import load_mission_file

        mission = load_mission_file(str(config_dir / "example_mission.yaml"))
        assert mission.name == "irchel_survey"
        assert mission.item_types[0] is ItemType.TAKEOFF
        assert mission.item_types[-1] is ItemType.RTL

    def test_example_mission_expands_to_many_waypoints(self, config_dir):
        from drone_bringup.core.mission import load_mission_file

        mission = load_mission_file(str(config_dir / "example_mission.yaml"))
        assert len(mission.expand()) > 20

    def test_example_mission_altitudes_are_relative(self, config_dir):
        from drone_bringup.core.mission import load_mission_file

        mission = load_mission_file(str(config_dir / "example_mission.yaml"))
        for waypoint in mission.expand():
            _, _, up = mission.waypoint_to_enu(waypoint)
            assert 0.0 <= up <= 120.0
