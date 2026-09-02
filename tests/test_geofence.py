"""Tests for polygon geometry, zone evaluation, and geofence loading."""

import math

import pytest

from drone_bringup.core.geofence import (
    Geofence,
    GeofenceError,
    GeofenceZone,
    Polygon2D,
    ZoneKind,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
)

SQUARE = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


class TestPointInPolygon:
    """Even-odd ray casting."""

    def test_centre_is_inside(self):
        assert point_in_polygon(50.0, 50.0, SQUARE)

    def test_outside_on_every_side(self):
        assert not point_in_polygon(-1.0, 50.0, SQUARE)
        assert not point_in_polygon(101.0, 50.0, SQUARE)
        assert not point_in_polygon(50.0, -1.0, SQUARE)
        assert not point_in_polygon(50.0, 101.0, SQUARE)

    def test_far_away_is_outside(self):
        assert not point_in_polygon(1e6, 1e6, SQUARE)

    def test_concave_notch_is_outside(self):
        # A U opening upwards: the middle of the notch must read as outside.
        u_shape = (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (70.0, 100.0),
            (70.0, 30.0),
            (30.0, 30.0),
            (30.0, 100.0),
            (0.0, 100.0),
        )
        assert point_in_polygon(50.0, 10.0, u_shape)
        assert not point_in_polygon(50.0, 70.0, u_shape)

    def test_winding_direction_does_not_matter(self):
        reversed_square = tuple(reversed(SQUARE))
        assert point_in_polygon(50.0, 50.0, reversed_square)


class TestPolygonMetrics:
    """Area and centroid."""

    def test_square_area(self):
        assert abs(polygon_area(SQUARE)) == pytest.approx(10000.0)

    def test_area_sign_follows_winding(self):
        assert polygon_area(SQUARE) > 0  # counter-clockwise
        assert polygon_area(tuple(reversed(SQUARE))) < 0

    def test_triangle_area(self):
        assert abs(polygon_area(((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))) == pytest.approx(50.0)

    def test_square_centroid(self):
        assert polygon_centroid(SQUARE) == pytest.approx((50.0, 50.0))

    def test_degenerate_polygon_falls_back_to_vertex_mean(self):
        collinear = ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
        assert polygon_centroid(collinear) == pytest.approx((10.0, 0.0))


class TestPolygon2D:
    """Distance, signed distance, and ray crossing."""

    @pytest.fixture
    def square(self):
        return Polygon2D(SQUARE)

    def test_rejects_two_vertices(self):
        with pytest.raises(GeofenceError, match="at least 3 vertices"):
            Polygon2D(((0.0, 0.0), (1.0, 1.0)))

    def test_distance_to_nearest_edge(self, square):
        assert square.distance_to_boundary(10.0, 50.0) == pytest.approx(10.0)
        assert square.distance_to_boundary(50.0, 95.0) == pytest.approx(5.0)

    def test_distance_from_outside(self, square):
        assert square.distance_to_boundary(-20.0, 50.0) == pytest.approx(20.0)

    def test_distance_to_a_corner(self, square):
        assert square.distance_to_boundary(-3.0, -4.0) == pytest.approx(5.0)

    def test_signed_distance_is_positive_inside(self, square):
        assert square.signed_distance(50.0, 50.0) == pytest.approx(50.0)

    def test_signed_distance_is_negative_outside(self, square):
        assert square.signed_distance(120.0, 50.0) == pytest.approx(-20.0)

    def test_area_and_centroid_properties(self, square):
        assert square.area == pytest.approx(10000.0)
        assert square.centroid == pytest.approx((50.0, 50.0))

    def test_time_to_cross_from_inside(self, square):
        # 40 m from the east edge at 10 m/s east: 4 s.
        assert square.time_to_cross(60.0, 50.0, 10.0, 0.0) == pytest.approx(4.0)

    def test_time_to_cross_picks_the_nearest_edge(self, square):
        # Heading north-east from the centre; the north edge at 50 m and the
        # east edge at 50 m are both hit at the same time, so t = 50/v.
        t = square.time_to_cross(50.0, 50.0, 10.0, 10.0)
        assert t == pytest.approx(5.0)

    def test_time_to_cross_from_outside_entering(self, square):
        assert square.time_to_cross(-30.0, 50.0, 5.0, 0.0) == pytest.approx(6.0)

    def test_time_to_cross_returns_none_when_stationary(self, square):
        assert square.time_to_cross(50.0, 50.0, 0.0, 0.0) is None

    def test_time_to_cross_returns_none_when_heading_away(self, square):
        assert square.time_to_cross(-30.0, 50.0, -5.0, 0.0) is None


class TestGeofenceAltitude:
    """The global altitude band."""

    @pytest.fixture
    def fence(self, origin):
        return Geofence(origin=origin, max_altitude_m=120.0, min_altitude_m=-10.0)

    def test_inside_the_band(self, fence):
        assert fence.check_local(0.0, 0.0, 50.0).inside

    def test_above_the_ceiling(self, fence):
        status = fence.check_local(0.0, 0.0, 130.0)
        assert status.breached
        assert "exceeds ceiling" in status.violations[0]

    def test_below_the_floor(self, fence):
        status = fence.check_local(0.0, 0.0, -20.0)
        assert status.breached
        assert "below floor" in status.violations[0]

    def test_margin_is_the_distance_to_the_nearer_limit(self, fence):
        assert fence.check_local(0.0, 0.0, 100.0).margin_m == pytest.approx(20.0)
        assert fence.check_local(0.0, 0.0, 0.0).margin_m == pytest.approx(10.0)

    def test_climb_predicts_a_ceiling_breach(self, fence):
        status = fence.check_local(0.0, 0.0, 100.0, vel_up=4.0)
        assert status.predicted_breach is not None
        assert status.predicted_breach.time_to_breach_s == pytest.approx(5.0)
        assert status.predicted_breach.kind.value == "altitude_ceiling"

    def test_descent_predicts_a_floor_breach(self, fence):
        status = fence.check_local(0.0, 0.0, 0.0, vel_up=-2.0)
        assert status.predicted_breach is not None
        assert status.predicted_breach.time_to_breach_s == pytest.approx(5.0)
        assert status.predicted_breach.kind.value == "altitude_floor"

    def test_level_flight_predicts_nothing(self, fence):
        assert fence.check_local(0.0, 0.0, 50.0, vel_up=0.0).predicted_breach is None

    def test_horizon_suppresses_a_distant_prediction(self, fence):
        # 70 m of headroom at 0.1 m/s is 700 s away; a 30 s horizon ignores it.
        status = fence.check_local(0.0, 0.0, 50.0, vel_up=0.1, horizon_s=30.0)
        assert status.predicted_breach is None

    def test_inverted_band_is_rejected(self, origin):
        with pytest.raises(GeofenceError, match="must exceed"):
            Geofence(origin=origin, max_altitude_m=10.0, min_altitude_m=50.0)


class TestInclusionZone:
    """Inclusion zones: the vehicle must stay inside."""

    @pytest.fixture
    def fence(self, origin):
        zone = GeofenceZone("area", ZoneKind.INCLUSION, Polygon2D(SQUARE))
        return Geofence(origin=origin, zones=[zone])

    def test_inside_passes(self, fence):
        assert fence.check_local(50.0, 50.0, 20.0).inside

    def test_outside_fails_with_a_distance(self, fence):
        status = fence.check_local(130.0, 50.0, 20.0)
        assert status.breached
        assert "outside inclusion zone 'area' by 30.0 m" in status.violations[0]

    def test_margin_shrinks_towards_the_edge(self, fence):
        assert fence.check_local(50.0, 50.0, 20.0).margin_m == pytest.approx(50.0)
        assert fence.check_local(95.0, 50.0, 20.0).margin_m == pytest.approx(5.0)

    def test_margin_is_negative_outside(self, fence):
        assert fence.check_local(130.0, 50.0, 20.0).margin_m == pytest.approx(-30.0)

    def test_exit_is_predicted(self, fence):
        status = fence.check_local(50.0, 50.0, 20.0, vel_east=10.0)
        assert status.predicted_breach is not None
        assert status.predicted_breach.kind.value == "inclusion_exit"
        assert status.predicted_breach.time_to_breach_s == pytest.approx(5.0)

    def test_time_to_breach_helper(self, fence):
        assert fence.time_to_breach(50.0, 50.0, 20.0, 0.0, 25.0, 0.0) == pytest.approx(2.0)

    def test_time_to_breach_is_none_when_hovering(self, fence):
        assert fence.time_to_breach(50.0, 50.0, 20.0, 0.0, 0.0, 0.0) is None

    def test_faster_flight_means_less_warning(self, fence):
        slow = fence.time_to_breach(50.0, 50.0, 20.0, 2.0, 0.0, 0.0)
        fast = fence.time_to_breach(50.0, 50.0, 20.0, 20.0, 0.0, 0.0)
        assert fast == pytest.approx(slow / 10.0)


class TestExclusionZone:
    """Exclusion zones: the vehicle must stay out."""

    @pytest.fixture
    def fence(self, origin):
        keepout = GeofenceZone("mast", ZoneKind.EXCLUSION, Polygon2D(SQUARE))
        return Geofence(origin=origin, zones=[keepout])

    def test_outside_passes(self, fence):
        assert fence.check_local(200.0, 200.0, 20.0).inside

    def test_inside_fails(self, fence):
        status = fence.check_local(50.0, 50.0, 20.0)
        assert status.breached
        assert "inside exclusion zone 'mast'" in status.violations[0]

    def test_margin_is_the_distance_to_the_keepout(self, fence):
        assert fence.check_local(-25.0, 50.0, 20.0).margin_m == pytest.approx(25.0)

    def test_entry_is_predicted(self, fence):
        status = fence.check_local(-40.0, 50.0, 20.0, vel_east=8.0)
        assert status.predicted_breach is not None
        assert status.predicted_breach.kind.value == "exclusion_entry"
        assert status.predicted_breach.time_to_breach_s == pytest.approx(5.0)

    def test_flying_away_predicts_nothing(self, fence):
        assert fence.check_local(-40.0, 50.0, 20.0, vel_east=-8.0).predicted_breach is None

    def test_zone_altitude_band_deactivates_the_zone(self, origin):
        # A keepout that only applies below 60 m: overflying it at 80 m is legal.
        zone = GeofenceZone(
            "mast", ZoneKind.EXCLUSION, Polygon2D(SQUARE), max_alt_m=60.0
        )
        fence = Geofence(origin=origin, zones=[zone])
        assert fence.check_local(50.0, 50.0, 80.0).inside
        assert fence.check_local(50.0, 50.0, 40.0).breached


class TestCombinedZones:
    """Inclusion and exclusion together, plus the soonest-breach selection."""

    @pytest.fixture
    def fence(self, origin):
        outer = GeofenceZone(
            "area",
            ZoneKind.INCLUSION,
            Polygon2D(((0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0))),
        )
        inner = GeofenceZone(
            "mast",
            ZoneKind.EXCLUSION,
            Polygon2D(((100.0, 100.0), (140.0, 100.0), (140.0, 140.0), (100.0, 140.0))),
        )
        return Geofence(origin=origin, zones=[outer, inner], max_altitude_m=120.0)

    def test_valid_position(self, fence):
        assert fence.check_local(50.0, 50.0, 40.0).inside

    def test_two_violations_are_reported_together(self, fence):
        # Outside the inclusion area AND above the ceiling.
        status = fence.check_local(400.0, 50.0, 200.0)
        assert len(status.violations) == 2

    def test_soonest_breach_wins(self, fence):
        # Heading east from x=60: exclusion entry at x=100 (40 m) comes before
        # the inclusion exit at x=300 (240 m).
        status = fence.check_local(60.0, 120.0, 40.0, vel_east=10.0)
        assert status.predicted_breach is not None
        assert status.predicted_breach.zone_name == "mast"
        assert status.predicted_breach.time_to_breach_s == pytest.approx(4.0)

    def test_ceiling_can_beat_a_horizontal_breach(self, fence):
        # 20 m of headroom at 10 m/s = 2 s, versus 240 m of horizontal room.
        status = fence.check_local(60.0, 250.0, 100.0, vel_east=10.0, vel_up=10.0)
        assert status.predicted_breach is not None
        assert status.predicted_breach.kind.value == "altitude_ceiling"
        assert status.predicted_breach.time_to_breach_s == pytest.approx(2.0)

    def test_zone_accessors(self, fence):
        assert [z.name for z in fence.inclusion_zones] == ["area"]
        assert [z.name for z in fence.exclusion_zones] == ["mast"]

    def test_duplicate_zone_names_are_rejected(self, origin):
        a = GeofenceZone("dup", ZoneKind.INCLUSION, Polygon2D(SQUARE))
        b = GeofenceZone("dup", ZoneKind.EXCLUSION, Polygon2D(SQUARE))
        with pytest.raises(GeofenceError, match="duplicate zone names"):
            Geofence(origin=origin, zones=[a, b])


class TestFromDict:
    """Loading a fence from the YAML-shaped dict."""

    def _doc(self, **overrides):
        doc = {
            "origin": {"latitude": 47.397742, "longitude": 8.545594, "altitude": 488.0},
            "max_altitude_m": 120.0,
            "zones": [
                {
                    "name": "area",
                    "kind": "inclusion",
                    "vertices": [
                        [47.3970, 8.5450],
                        [47.3970, 8.5470],
                        [47.3990, 8.5470],
                        [47.3990, 8.5450],
                    ],
                }
            ],
        }
        doc.update(overrides)
        return doc

    def test_loads_a_valid_document(self):
        fence = Geofence.from_dict(self._doc())
        assert len(fence.zones) == 1
        assert fence.zones[0].kind is ZoneKind.INCLUSION
        assert fence.max_altitude_m == pytest.approx(120.0)

    def test_vertices_are_projected_into_metres(self):
        fence = Geofence.from_dict(self._doc())
        # Roughly 0.002 deg lat x 0.002 deg lon at 47.4: ~222 m x ~151 m.
        assert fence.zones[0].polygon.area == pytest.approx(222 * 151, rel=0.05)

    def test_root_must_be_a_mapping(self):
        with pytest.raises(GeofenceError, match="root must be a mapping"):
            Geofence.from_dict([])  # type: ignore[arg-type]

    def test_missing_origin(self):
        doc = self._doc()
        del doc["origin"]
        with pytest.raises(GeofenceError, match="missing required 'origin'"):
            Geofence.from_dict(doc)

    def test_origin_missing_latitude(self):
        doc = self._doc(origin={"longitude": 8.5})
        with pytest.raises(GeofenceError, match="missing required key 'latitude'"):
            Geofence.from_dict(doc)

    def test_unknown_zone_kind(self):
        doc = self._doc()
        doc["zones"][0]["kind"] = "maybe"
        with pytest.raises(GeofenceError, match="kind must be one of"):
            Geofence.from_dict(doc)

    def test_too_few_vertices(self):
        doc = self._doc()
        doc["zones"][0]["vertices"] = [[47.397, 8.545], [47.398, 8.546]]
        with pytest.raises(GeofenceError, match="at least 3"):
            Geofence.from_dict(doc)

    def test_malformed_vertex(self):
        doc = self._doc()
        doc["zones"][0]["vertices"][1] = [47.397]
        with pytest.raises(GeofenceError, match=r"vertices\[1\] must be a \[lat, lon\] pair"):
            Geofence.from_dict(doc)

    def test_zones_must_be_a_list(self):
        with pytest.raises(GeofenceError, match="zones must be a list"):
            Geofence.from_dict(self._doc(zones={"a": 1}))


class TestGeodeticCheck:
    """The geodetic wrapper agrees with the local one."""

    def test_geodetic_matches_local(self, origin):
        zone = GeofenceZone("area", ZoneKind.INCLUSION, Polygon2D(SQUARE))
        fence = Geofence(origin=origin, zones=[zone])
        lat, lon, alt = origin.enu_to_geodetic(50.0, 50.0, 20.0)
        assert fence.check_geodetic(lat, lon, alt).inside
        assert fence.check_local(50.0, 50.0, 20.0).inside

    def test_geodetic_detects_a_breach(self, origin):
        zone = GeofenceZone("area", ZoneKind.INCLUSION, Polygon2D(SQUARE))
        fence = Geofence(origin=origin, zones=[zone])
        lat, lon, alt = origin.enu_to_geodetic(500.0, 50.0, 20.0)
        assert fence.check_geodetic(lat, lon, alt).breached


class TestShippedGeofence:
    """The example geofence in config/ must load and be consistent."""

    def test_example_geofence_loads(self, config_dir):
        import yaml

        with open(config_dir / "example_geofence.yaml", encoding="utf-8") as handle:
            fence = Geofence.from_dict(yaml.safe_load(handle))
        assert len(fence.inclusion_zones) == 1
        assert len(fence.exclusion_zones) == 1
        assert fence.max_altitude_m == pytest.approx(120.0)

    def test_example_mission_stays_inside_the_example_geofence(self, config_dir):
        import yaml

        from drone_bringup.core.mission import load_mission_file

        with open(config_dir / "example_geofence.yaml", encoding="utf-8") as handle:
            fence = Geofence.from_dict(yaml.safe_load(handle))
        mission = load_mission_file(str(config_dir / "example_mission.yaml"))
        for waypoint in mission.expand():
            east, north, up = mission.waypoint_to_enu(waypoint)
            status = fence.check_local(east, north, up)
            assert status.inside, f"{waypoint.label}: {status.violations}"

    def test_example_mission_keeps_a_real_margin(self, config_dir):
        import yaml

        from drone_bringup.core.mission import load_mission_file

        with open(config_dir / "example_geofence.yaml", encoding="utf-8") as handle:
            fence = Geofence.from_dict(yaml.safe_load(handle))
        mission = load_mission_file(str(config_dir / "example_mission.yaml"))
        margins = [
            fence.check_local(*mission.waypoint_to_enu(w)).margin_m
            for w in mission.expand()
        ]
        assert min(margins) > 1.0
        assert all(math.isfinite(m) for m in margins)
