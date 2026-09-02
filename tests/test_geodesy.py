"""Tests for WGS84 <-> local frame conversions.

Expectations are either hand-computed from the WGS84 defining constants or
derived from exact geometric identities (equator, poles, prime meridian), not
copied from another implementation.
"""

import math

import pytest

from drone_bringup.core.geodesy import (
    EARTH_MEAN_RADIUS,
    WGS84_A,
    WGS84_B,
    LocalOrigin,
    destination_point,
    ecef_to_geodetic,
    enu_to_ned,
    enu_yaw_to_ned_yaw,
    geodetic_to_ecef,
    haversine_distance,
    initial_bearing,
    ned_to_enu,
    ned_yaw_to_enu_yaw,
    wrap_2pi,
    wrap_pi,
)


class TestConstants:
    """The derived ellipsoid constants must follow from the defining ones."""

    def test_semi_minor_axis(self):
        # b = a(1 - f). WGS84 b is 6356752.314245 m to six decimal places.
        assert WGS84_B == pytest.approx(6356752.314245, abs=1e-6)

    def test_flattening_relationship(self):
        assert WGS84_A > WGS84_B
        assert (WGS84_A - WGS84_B) / WGS84_A == pytest.approx(1 / 298.257223563)


class TestEcef:
    """Geodetic <-> ECEF against points with exact closed forms."""

    def test_equator_prime_meridian(self):
        # lat=0, lon=0, alt=0 sits on the +x axis at exactly the semi-major axis.
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert x == pytest.approx(WGS84_A, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-9)
        assert z == pytest.approx(0.0, abs=1e-9)

    def test_equator_ninety_east(self):
        x, y, z = geodetic_to_ecef(0.0, 90.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(WGS84_A, abs=1e-6)
        assert z == pytest.approx(0.0, abs=1e-9)

    def test_north_pole(self):
        # At the pole the ECEF z is exactly the semi-minor axis.
        x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert math.hypot(x, y) == pytest.approx(0.0, abs=1e-6)
        assert z == pytest.approx(WGS84_B, abs=1e-6)

    def test_altitude_adds_along_the_normal_at_the_equator(self):
        x0, _, _ = geodetic_to_ecef(0.0, 0.0, 0.0)
        x1, _, _ = geodetic_to_ecef(0.0, 0.0, 1000.0)
        assert x1 - x0 == pytest.approx(1000.0, abs=1e-6)

    @pytest.mark.parametrize(
        "lat,lon,alt",
        [
            (0.0, 0.0, 0.0),
            (47.397742, 8.545594, 488.0),
            (-33.8688, 151.2093, 58.0),
            (89.0, -179.0, 12000.0),
            (-45.0, 179.999, -50.0),
        ],
    )
    def test_ecef_round_trip(self, lat, lon, alt):
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
        assert lat2 == pytest.approx(lat, abs=1e-9)
        assert lon2 == pytest.approx(lon, abs=1e-9)
        assert alt2 == pytest.approx(alt, abs=1e-4)


class TestEnuNed:
    """The ENU/NED clash: swap the first two axes, negate the third."""

    def test_enu_to_ned_mapping(self):
        assert enu_to_ned(1.0, 2.0, 3.0) == (2.0, 1.0, -3.0)

    def test_ned_to_enu_mapping(self):
        assert ned_to_enu(2.0, 1.0, -3.0) == (1.0, 2.0, 3.0)

    def test_conversion_is_an_involution(self):
        vec = (12.5, -3.25, 40.0)
        assert ned_to_enu(*enu_to_ned(*vec)) == vec
        assert enu_to_ned(*ned_to_enu(*vec)) == vec

    def test_naive_z_flip_is_not_the_same_thing(self):
        # The mistake this module exists to prevent: negating z alone.
        east, north, up = 10.0, 0.0, 5.0
        naive = (east, north, -up)
        assert enu_to_ned(east, north, up) != naive

    def test_enu_east_is_ned_east(self):
        # Pure east in ENU must be pure east (y) in NED, with no north component.
        n, e, d = enu_to_ned(7.0, 0.0, 0.0)
        assert (n, e, d) == (0.0, 7.0, 0.0)

    def test_enu_up_is_negative_ned_down(self):
        n, e, d = enu_to_ned(0.0, 0.0, 20.0)
        assert d == -20.0


class TestYawConventions:
    """ENU yaw (CCW from East) vs NED heading (CW from North)."""

    def test_east_in_enu_is_ninety_degrees_in_ned(self):
        assert math.degrees(enu_yaw_to_ned_yaw(0.0)) == pytest.approx(90.0)

    def test_north_in_enu_is_zero_in_ned(self):
        assert math.degrees(enu_yaw_to_ned_yaw(math.pi / 2)) == pytest.approx(0.0)

    def test_yaw_conversion_is_an_involution(self):
        for deg in (-170.0, -45.0, 0.0, 30.0, 90.0, 179.0):
            rad = math.radians(deg)
            assert ned_yaw_to_enu_yaw(enu_yaw_to_ned_yaw(rad)) == pytest.approx(rad)

    def test_result_is_wrapped_to_pi(self):
        for deg in range(-360, 361, 17):
            out = enu_yaw_to_ned_yaw(math.radians(deg))
            assert -math.pi < out <= math.pi + 1e-12


class TestWrapping:
    """Angle wrapping helpers."""

    def test_wrap_pi_keeps_pi(self):
        assert wrap_pi(math.pi) == pytest.approx(math.pi)

    def test_wrap_pi_maps_minus_pi_to_pi(self):
        # (-pi, pi] is half-open at the bottom.
        assert wrap_pi(-math.pi) == pytest.approx(math.pi)

    def test_wrap_pi_unwinds_multiple_turns(self):
        assert wrap_pi(3 * math.pi + 0.5) == pytest.approx(0.5 - math.pi, abs=1e-12)

    def test_wrap_2pi_range(self):
        assert wrap_2pi(-0.5) == pytest.approx(2 * math.pi - 0.5)
        assert 0.0 <= wrap_2pi(-7.0) < 2 * math.pi


class TestHaversine:
    """Spherical range and bearing helpers."""

    def test_quarter_circumference_equator_to_pole(self):
        expected = 0.5 * math.pi * EARTH_MEAN_RADIUS
        assert haversine_distance(0.0, 0.0, 90.0, 0.0) == pytest.approx(expected, rel=1e-12)

    def test_one_degree_of_latitude(self):
        # One degree of arc on the mean sphere: R * pi/180.
        expected = EARTH_MEAN_RADIUS * math.pi / 180.0
        assert haversine_distance(0.0, 0.0, 1.0, 0.0) == pytest.approx(expected, rel=1e-12)

    def test_zero_distance(self):
        assert haversine_distance(47.4, 8.5, 47.4, 8.5) == pytest.approx(0.0, abs=1e-9)

    def test_symmetry(self):
        a = haversine_distance(47.0, 8.0, 48.0, 9.0)
        b = haversine_distance(48.0, 9.0, 47.0, 8.0)
        assert a == pytest.approx(b)

    def test_bearing_due_north(self):
        assert initial_bearing(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_bearing_due_east(self):
        assert initial_bearing(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=1e-9)

    def test_bearing_due_west_is_270(self):
        # Range is [0, 360), so west is 270 and not -90.
        assert initial_bearing(0.0, 0.0, 0.0, -1.0) == pytest.approx(270.0, abs=1e-9)

    def test_destination_point_round_trip(self):
        lat, lon = destination_point(47.397742, 8.545594, 45.0, 1000.0)
        back = haversine_distance(47.397742, 8.545594, lat, lon)
        assert back == pytest.approx(1000.0, rel=1e-9)
        assert initial_bearing(47.397742, 8.545594, lat, lon) == pytest.approx(45.0, abs=1e-6)


class TestLocalOrigin:
    """Local tangent-plane projection."""

    def test_origin_projects_to_zero(self, origin):
        e, n, u = origin.geodetic_to_enu(
            origin.latitude_deg, origin.longitude_deg, origin.altitude_m
        )
        assert (e, n, u) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    def test_pure_altitude_change_is_pure_up(self, origin):
        e, n, u = origin.geodetic_to_enu(
            origin.latitude_deg, origin.longitude_deg, origin.altitude_m + 25.0
        )
        assert e == pytest.approx(0.0, abs=1e-6)
        assert n == pytest.approx(0.0, abs=1e-6)
        assert u == pytest.approx(25.0, abs=1e-6)

    def test_north_offset_has_no_east_component(self, origin):
        # Moving purely in latitude must produce purely +North in ENU.
        e, n, _ = origin.geodetic_to_enu(
            origin.latitude_deg + 0.001, origin.longitude_deg, origin.altitude_m
        )
        assert abs(e) < 1e-6
        assert n > 100.0  # 0.001 deg of latitude is ~111 m

    def test_one_milli_degree_of_latitude_is_about_111_m(self, origin):
        _, n, _ = origin.geodetic_to_enu(
            origin.latitude_deg + 0.001, origin.longitude_deg, origin.altitude_m
        )
        # Meridian arc at 47.4 deg latitude: ~111.19 m per milli-degree.
        assert n == pytest.approx(111.2, abs=0.5)

    def test_longitude_scale_shrinks_with_latitude(self, origin):
        _, _, _ = origin.geodetic_to_enu(0, 0, 0)  # exercise the cached trig path
        e, _, _ = origin.geodetic_to_enu(
            origin.latitude_deg, origin.longitude_deg + 0.001, origin.altitude_m
        )
        # 0.001 deg of longitude at 47.4 deg: ~111.3 * cos(47.4) = ~75.4 m.
        assert e == pytest.approx(75.4, abs=1.0)

    @pytest.mark.parametrize(
        "enu",
        [
            (0.0, 0.0, 0.0),
            (100.0, 200.0, 30.0),
            (-1500.0, 900.0, -45.0),
            (5.0, -5.0, 0.5),
        ],
    )
    def test_enu_round_trip(self, origin, enu):
        lat, lon, alt = origin.enu_to_geodetic(*enu)
        back = origin.geodetic_to_enu(lat, lon, alt)
        assert back == pytest.approx(enu, abs=1e-6)

    @pytest.mark.parametrize(
        "ned",
        [
            (0.0, 0.0, 0.0),
            (200.0, 100.0, -30.0),
            (-750.0, 1200.0, 15.0),
        ],
    )
    def test_ned_round_trip(self, origin, ned):
        lat, lon, alt = origin.ned_to_geodetic(*ned)
        back = origin.geodetic_to_ned(lat, lon, alt)
        assert back == pytest.approx(ned, abs=1e-6)

    def test_enu_and_ned_agree(self, origin):
        lat, lon, alt = origin.enu_to_geodetic(120.0, -60.0, 25.0)
        e, n, u = origin.geodetic_to_enu(lat, lon, alt)
        north, east, down = origin.geodetic_to_ned(lat, lon, alt)
        assert (north, east, down) == pytest.approx(enu_to_ned(e, n, u), abs=1e-9)

    def test_ground_distance_ignores_altitude(self, origin):
        lat, lon, _ = origin.enu_to_geodetic(300.0, 400.0, 0.0)
        assert origin.ground_distance_to(lat, lon) == pytest.approx(500.0, abs=1e-3)

    def test_distance_to_includes_altitude(self, origin):
        lat, lon, alt = origin.enu_to_geodetic(3.0, 4.0, 12.0)
        assert origin.distance_to(lat, lon, alt) == pytest.approx(13.0, abs=1e-6)

    def test_haversine_agrees_with_projection_at_short_range(self, origin):
        lat, lon, _ = origin.enu_to_geodetic(150.0, 250.0, 0.0)
        projected = origin.ground_distance_to(lat, lon)
        spherical = haversine_distance(
            origin.latitude_deg, origin.longitude_deg, lat, lon
        )
        # Sphere vs ellipsoid: sub-0.5% over a few hundred metres.
        assert spherical == pytest.approx(projected, rel=0.005)

    def test_rejects_out_of_range_latitude(self):
        with pytest.raises(ValueError, match="latitude_deg"):
            LocalOrigin(91.0, 0.0, 0.0)

    def test_rejects_out_of_range_longitude(self):
        with pytest.raises(ValueError, match="longitude_deg"):
            LocalOrigin(0.0, 181.0, 0.0)

    def test_origin_is_hashable_and_frozen(self, origin):
        # Frozen so it can be shared between nodes without anyone re-anchoring it.
        assert hash(origin) == hash(LocalOrigin(*(47.397742, 8.545594, 488.0)))
        with pytest.raises(Exception):
            origin.latitude_deg = 0.0  # type: ignore[misc]
