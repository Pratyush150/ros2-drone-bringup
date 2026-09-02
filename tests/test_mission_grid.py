"""Tests for lawnmower/survey grid generation.

The two properties that matter for a survey are **coverage** (no point in the
polygon is further from a flight line than half the swath) and **ordering**
(consecutive lines alternate direction, so the vehicle does not deadhead back
across the field between every pass). Both are asserted here on real geometry,
not on a waypoint count.
"""

import math

import pytest

from drone_bringup.core.geodesy import LocalOrigin
from drone_bringup.core.geofence import point_in_polygon
from drone_bringup.core.mission import (
    MissionValidationError,
    OrbitItem,
    generate_lawnmower,
)


def rect_polygon(origin: LocalOrigin, width_e: float, height_n: float):
    """A rectangle in geodetic coordinates, given local ENU extents in metres."""
    corners = [(0.0, 0.0), (width_e, 0.0), (width_e, height_n), (0.0, height_n)]
    return [origin.enu_to_geodetic(e, n, 0.0)[:2] for e, n in corners]


def to_local(origin: LocalOrigin, waypoints):
    """Project waypoints back into local ENU (east, north) pairs."""
    return [
        origin.geodetic_to_enu(w.latitude, w.longitude, origin.altitude_m)[:2]
        for w in waypoints
    ]


def segment_distance(px, py, ax, ay, bx, by):
    """Distance from a point to a segment; duplicated here so the test is independent."""
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class TestBasicShape:
    """Waypoint count, altitude, and speed propagation."""

    def test_generates_pairs_of_waypoints(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        waypoints = generate_lawnmower(poly, origin, 40.0, 25.0, heading_deg=0.0)
        assert len(waypoints) % 2 == 0
        assert len(waypoints) >= 2

    def test_line_count_matches_spacing(self, origin):
        # 100 m of extent across the lines at 25 m spacing gives 4 lines.
        poly = rect_polygon(origin, 200.0, 100.0)
        waypoints = generate_lawnmower(poly, origin, 40.0, 25.0, heading_deg=90.0)
        assert len(waypoints) == 8  # 4 lines x 2 endpoints

    def test_halving_the_spacing_roughly_doubles_the_lines(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        coarse = generate_lawnmower(poly, origin, 40.0, 25.0, heading_deg=90.0)
        fine = generate_lawnmower(poly, origin, 40.0, 12.5, heading_deg=90.0)
        assert len(fine) == pytest.approx(2 * len(coarse), abs=2)

    def test_altitude_is_applied_to_every_waypoint(self, origin):
        poly = rect_polygon(origin, 120.0, 120.0)
        waypoints = generate_lawnmower(poly, origin, 55.0, 30.0)
        assert all(w.altitude == pytest.approx(55.0) for w in waypoints)

    def test_speed_and_radius_are_applied(self, origin):
        poly = rect_polygon(origin, 120.0, 120.0)
        waypoints = generate_lawnmower(
            poly, origin, 40.0, 30.0, speed=6.5, acceptance_radius=4.0
        )
        assert all(w.speed == pytest.approx(6.5) for w in waypoints)
        assert all(w.acceptance_radius == pytest.approx(4.0) for w in waypoints)

    def test_labels_identify_the_line(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        waypoints = generate_lawnmower(
            poly, origin, 40.0, 25.0, heading_deg=90.0, label_prefix="s3"
        )
        assert waypoints[0].label == "s3_l0"
        assert waypoints[-1].label == "s3_l3"


class TestHeading:
    """The heading argument must actually orient the lines."""

    def test_heading_zero_gives_north_south_lines(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(origin, generate_lawnmower(poly, origin, 40.0, 25.0, 0.0))
        for a, b in zip(local[::2], local[1::2]):
            # Endpoints of one line share an east coordinate and differ in north.
            assert a[0] == pytest.approx(b[0], abs=1e-6)
            assert abs(a[1] - b[1]) == pytest.approx(100.0, abs=1e-3)

    def test_heading_ninety_gives_east_west_lines(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(origin, generate_lawnmower(poly, origin, 40.0, 25.0, 90.0))
        for a, b in zip(local[::2], local[1::2]):
            assert a[1] == pytest.approx(b[1], abs=1e-6)
            # 2 cm tolerance: the polygon round-trips ENU -> WGS84 -> ENU, and
            # the tangent-plane approximation costs about a millimetre per
            # hundred metres.
            assert abs(a[0] - b[0]) == pytest.approx(200.0, abs=0.02)

    def test_heading_changes_the_line_direction(self, origin):
        poly = rect_polygon(origin, 200.0, 200.0)
        diag = to_local(origin, generate_lawnmower(poly, origin, 40.0, 40.0, 45.0))
        # A 45 degree line has equal |de| and |dn|.
        a, b = diag[0], diag[1]
        assert abs(abs(b[0] - a[0]) - abs(b[1] - a[1])) < 1e-6

    def test_line_spacing_is_the_requested_spacing(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(origin, generate_lawnmower(poly, origin, 40.0, 25.0, 90.0))
        norths = [a[1] for a in local[::2]]
        gaps = [abs(b - a) for a, b in zip(norths, norths[1:])]
        assert all(g == pytest.approx(25.0, abs=1e-3) for g in gaps)


class TestCoverage:
    """Every interior point must be within half a swath of some flight line."""

    @pytest.mark.parametrize("heading", [0.0, 45.0, 90.0, 135.0, 210.0])
    def test_rectangle_is_covered(self, origin, heading):
        spacing = 20.0
        poly = rect_polygon(origin, 180.0, 140.0)
        waypoints = generate_lawnmower(poly, origin, 40.0, spacing, heading)
        local = to_local(origin, waypoints)
        lines = list(zip(local[::2], local[1::2]))
        assert lines

        worst = 0.0
        for i in range(1, 18):
            for j in range(1, 14):
                px, py = i * 10.0, j * 10.0
                nearest = min(
                    segment_distance(px, py, a[0], a[1], b[0], b[1]) for a, b in lines
                )
                worst = max(worst, nearest)
        # Half the spacing plus a little slack for the projection round trip.
        assert worst <= spacing / 2.0 + 0.5

    def test_all_waypoints_lie_on_or_inside_the_polygon(self, origin):
        poly = rect_polygon(origin, 180.0, 140.0)
        waypoints = generate_lawnmower(poly, origin, 40.0, 20.0, 90.0)
        local_poly = [
            origin.geodetic_to_enu(lat, lon, 0.0)[:2] for lat, lon in poly
        ]
        for east, north in to_local(origin, waypoints):
            # Endpoints sit exactly on the boundary, so allow a small tolerance
            # by nudging inward before the containment test.
            cx = sum(p[0] for p in local_poly) / len(local_poly)
            cy = sum(p[1] for p in local_poly) / len(local_poly)
            nudged = (east + (cx - east) * 1e-3, north + (cy - north) * 1e-3)
            assert point_in_polygon(nudged[0], nudged[1], local_poly)

    def test_concave_polygon_is_filled_in_separate_segments(self, origin):
        # A U-shape. A scanline through the notch must produce two separate
        # inside-segments rather than one line spanning the gap -- a bounding-box
        # "grid generator" would fly straight through the middle of the U.
        corners = [
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 160.0),
            (140.0, 160.0),
            (140.0, 60.0),
            (60.0, 60.0),
            (60.0, 160.0),
            (0.0, 160.0),
        ]
        poly = [origin.enu_to_geodetic(e, n, 0.0)[:2] for e, n in corners]
        waypoints = generate_lawnmower(poly, origin, 40.0, 20.0, heading_deg=90.0)
        local = to_local(origin, waypoints)
        lines = list(zip(local[::2], local[1::2]))

        below = [ln for ln in lines if ln[0][1] < 55.0]
        above = [ln for ln in lines if ln[0][1] > 65.0]
        assert below and above
        # Below the notch the polygon is solid: one 200 m line per scan.
        assert all(abs(a[0] - b[0]) == pytest.approx(200.0, abs=0.05) for a, b in below)
        # Above it each scan is split into two 60 m arms, so twice as many
        # segments and none of them spans the 80 m notch.
        assert all(abs(a[0] - b[0]) == pytest.approx(60.0, abs=0.05) for a, b in above)
        assert len(above) == 2 * len({round(ln[0][1], 3) for ln in above})

    def test_concave_polygon_never_flies_through_the_notch(self, origin):
        corners = [
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 160.0),
            (140.0, 160.0),
            (140.0, 60.0),
            (60.0, 60.0),
            (60.0, 160.0),
            (0.0, 160.0),
        ]
        poly = [origin.enu_to_geodetic(e, n, 0.0)[:2] for e, n in corners]
        local_poly = [origin.geodetic_to_enu(a, b, 0.0)[:2] for a, b in poly]
        waypoints = generate_lawnmower(poly, origin, 40.0, 20.0, heading_deg=90.0)
        local = to_local(origin, waypoints)
        for a, b in zip(local[::2], local[1::2]):
            midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            assert point_in_polygon(midpoint[0], midpoint[1], local_poly)

    def test_triangle_lines_shrink_towards_the_apex(self, origin):
        corners = [(0.0, 0.0), (200.0, 0.0), (100.0, 150.0)]
        poly = [origin.enu_to_geodetic(e, n, 0.0)[:2] for e, n in corners]
        waypoints = generate_lawnmower(poly, origin, 40.0, 20.0, heading_deg=90.0)
        local = to_local(origin, waypoints)
        lengths = [abs(b[0] - a[0]) for a, b in zip(local[::2], local[1::2])]
        assert len(lengths) >= 3
        # Monotonically shorter as the scan climbs towards the apex.
        assert all(b <= a + 1e-6 for a, b in zip(lengths, lengths[1:]))


class TestOrdering:
    """Boustrophedon ordering: alternate direction, no deadhead legs."""

    def test_consecutive_lines_alternate_direction(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(origin, generate_lawnmower(poly, origin, 40.0, 25.0, 90.0))
        directions = [
            math.copysign(1.0, b[0] - a[0]) for a, b in zip(local[::2], local[1::2])
        ]
        assert len(directions) >= 3
        assert all(a != b for a, b in zip(directions, directions[1:]))

    def test_turn_legs_are_the_line_spacing_not_the_line_length(self, origin):
        # The whole point of alternating: the hop between the end of one line
        # and the start of the next is one spacing, not one full line.
        spacing = 25.0
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(origin, generate_lawnmower(poly, origin, 40.0, spacing, 90.0))
        for i in range(1, len(local) - 1, 2):
            end = local[i]
            nxt = local[i + 1]
            assert math.dist(end, nxt) == pytest.approx(spacing, abs=1e-3)

    def test_a_naive_same_direction_ordering_would_be_much_longer(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(origin, generate_lawnmower(poly, origin, 40.0, 25.0, 90.0))
        actual = sum(math.dist(a, b) for a, b in zip(local, local[1:]))
        # Same lines, but every one flown left-to-right: each turn costs a full
        # line length instead of one spacing.
        naive_lines = [
            (min(a, b), max(a, b)) for a, b in zip(local[::2], local[1::2])
        ]
        flat = [p for line in naive_lines for p in line]
        naive = sum(math.dist(a, b) for a, b in zip(flat, flat[1:]))
        assert actual < naive


class TestLineExtension:
    """Overshoot at both ends of every line."""

    def test_extension_lengthens_each_line_by_twice_the_value(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        plain = to_local(origin, generate_lawnmower(poly, origin, 40.0, 25.0, 90.0))
        extended = to_local(
            origin,
            generate_lawnmower(poly, origin, 40.0, 25.0, 90.0, line_extension=15.0),
        )
        plain_len = abs(plain[1][0] - plain[0][0])
        ext_len = abs(extended[1][0] - extended[0][0])
        assert ext_len - plain_len == pytest.approx(30.0, abs=1e-3)

    def test_extension_does_not_change_the_line_count(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        plain = generate_lawnmower(poly, origin, 40.0, 25.0, 90.0)
        extended = generate_lawnmower(poly, origin, 40.0, 25.0, 90.0, line_extension=15.0)
        assert len(plain) == len(extended)

    def test_extension_preserves_direction(self, origin):
        poly = rect_polygon(origin, 200.0, 100.0)
        local = to_local(
            origin,
            generate_lawnmower(poly, origin, 40.0, 25.0, 90.0, line_extension=10.0),
        )
        directions = [
            math.copysign(1.0, b[0] - a[0]) for a, b in zip(local[::2], local[1::2])
        ]
        assert all(a != b for a, b in zip(directions, directions[1:]))


class TestGridErrors:
    """Bad inputs must be rejected with a useful message."""

    def test_too_few_vertices(self, origin):
        with pytest.raises(MissionValidationError, match="at least 3 vertices"):
            generate_lawnmower(
                [(47.4, 8.5), (47.41, 8.5)], origin, 40.0, 20.0
            )

    def test_zero_spacing(self, origin):
        poly = rect_polygon(origin, 100.0, 100.0)
        with pytest.raises(MissionValidationError, match="spacing must be > 0"):
            generate_lawnmower(poly, origin, 40.0, 0.0)

    def test_negative_line_extension(self, origin):
        poly = rect_polygon(origin, 100.0, 100.0)
        with pytest.raises(MissionValidationError, match="line_extension must be >= 0"):
            generate_lawnmower(poly, origin, 40.0, 20.0, line_extension=-1.0)

    def test_degenerate_polygon(self, origin):
        # Three collinear points have no extent across any heading.
        poly = [origin.enu_to_geodetic(e, 0.0, 0.0)[:2] for e in (0.0, 50.0, 100.0)]
        with pytest.raises(MissionValidationError, match="degenerate"):
            generate_lawnmower(poly, origin, 40.0, 20.0, heading_deg=90.0)

    def test_spacing_larger_than_the_polygon_gives_one_centre_line(self, origin):
        poly = rect_polygon(origin, 200.0, 30.0)
        waypoints = generate_lawnmower(poly, origin, 40.0, 500.0, heading_deg=90.0)
        local = to_local(origin, waypoints)
        assert len(local) == 2
        assert local[0][1] == pytest.approx(15.0, abs=1e-3)  # centred in the 30 m extent


class TestOrbitExpansion:
    """Orbit items discretise into a ring at the requested radius."""

    def _mission(self, origin):
        from drone_bringup.core.mission import Mission, TakeoffItem

        return Mission(name="t", origin=origin, items=[TakeoffItem(index=0, altitude=10.0)])

    def test_orbit_points_sit_on_the_circle(self, origin):
        mission = self._mission(origin)
        centre_lat, centre_lon, _ = origin.enu_to_geodetic(100.0, 50.0, 0.0)
        item = OrbitItem(
            index=1,
            latitude=centre_lat,
            longitude=centre_lon,
            altitude=30.0,
            radius=25.0,
            turns=1.0,
            points_per_turn=12,
        )
        for waypoint in item.expand(mission):
            e, n, _ = origin.geodetic_to_enu(
                waypoint.latitude, waypoint.longitude, origin.altitude_m
            )
            assert math.hypot(e - 100.0, n - 50.0) == pytest.approx(25.0, abs=1e-3)

    def test_orbit_produces_one_extra_point_to_close_the_ring(self, origin):
        mission = self._mission(origin)
        item = OrbitItem(
            index=1,
            latitude=origin.latitude_deg,
            longitude=origin.longitude_deg,
            altitude=30.0,
            radius=20.0,
            turns=1.0,
            points_per_turn=12,
        )
        assert len(item.expand(mission)) == 13

    def test_two_turns_double_the_points(self, origin):
        mission = self._mission(origin)
        one = OrbitItem(
            index=1,
            latitude=origin.latitude_deg,
            longitude=origin.longitude_deg,
            altitude=30.0,
            radius=20.0,
            turns=1.0,
            points_per_turn=12,
        ).expand(mission)
        two = OrbitItem(
            index=1,
            latitude=origin.latitude_deg,
            longitude=origin.longitude_deg,
            altitude=30.0,
            radius=20.0,
            turns=2.0,
            points_per_turn=12,
        ).expand(mission)
        assert len(two) == 2 * len(one) - 1

    def test_orbit_yaw_points_at_the_centre(self, origin):
        mission = self._mission(origin)
        item = OrbitItem(
            index=1,
            latitude=origin.latitude_deg,
            longitude=origin.longitude_deg,
            altitude=30.0,
            radius=20.0,
            turns=1.0,
            points_per_turn=4,
        )
        for waypoint in item.expand(mission):
            assert waypoint.yaw_deg is not None
            e, n, _ = origin.geodetic_to_enu(
                waypoint.latitude, waypoint.longitude, origin.altitude_m
            )
            # The stored yaw is an ENU angle from the point back to the centre.
            # Compare as a wrapped difference: 0 and 360 are the same heading,
            # and a naive modulo comparison fails on exactly that boundary.
            expected = math.degrees(math.atan2(-n, -e))
            delta = (waypoint.yaw_deg - expected + 180.0) % 360.0 - 180.0
            assert delta == pytest.approx(0.0, abs=1e-6)

    def test_clockwise_and_counter_clockwise_differ(self, origin):
        mission = self._mission(origin)
        common = dict(
            index=1,
            latitude=origin.latitude_deg,
            longitude=origin.longitude_deg,
            altitude=30.0,
            radius=20.0,
            turns=1.0,
            points_per_turn=8,
        )
        cw = OrbitItem(clockwise=True, **common).expand(mission)
        ccw = OrbitItem(clockwise=False, **common).expand(mission)
        assert cw[1].latitude != pytest.approx(ccw[1].latitude)
