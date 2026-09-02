"""WGS84 <-> local tangent-plane conversions, with an explicit ENU/NED split.

Why this module exists
----------------------
Every drone project eventually needs "how far is that waypoint, in metres, from
where I am now". You get there by projecting WGS84 geodetic coordinates onto a
local tangent plane anchored at a fixed origin. Two things bite people:

1. **ENU vs NED.** ROS (REP-103) is East-North-Up, right-handed, x=East,
   y=North, z=Up. PX4 internally, MAVLink ``LOCAL_POSITION_NED``, and basically
   all of aviation are North-East-Down, x=North, y=East, z=Down. The two are
   *not* related by a sign flip on one axis; the correct mapping is

       ned = (enu.y, enu.x, -enu.z)
       enu = (ned.y, ned.x, -ned.z)

   i.e. swap the first two components and negate the third. That operation is
   its own inverse. If you "convert" by only negating z you silently get a
   left-handed frame and every yaw you compute is mirrored. MAVROS already does
   this swap for you on its topics, which is exactly why mixing a raw MAVLink
   stream with a MAVROS topic in the same node produces a drone that flies 90
   degrees off heading.

2. **Yaw.** ENU yaw is measured counter-clockwise from East. NED yaw (aviation
   heading) is measured clockwise from North. The conversion is
   ``yaw_ned = pi/2 - yaw_enu`` (wrapped), and again it is its own inverse.
   See :func:`enu_yaw_to_ned_yaw`.

The projection itself is a standard geodetic->ECEF->ENU chain, which is exact
to floating point for the ECEF step and correct for any local-tangent distance
you would fly a multirotor over. For quick range/bearing checks over a sphere
there is also :func:`haversine_distance` and :func:`initial_bearing`.

Pure Python, no ROS, no numpy. Import it anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "WGS84_A",
    "WGS84_F",
    "WGS84_B",
    "WGS84_E_SQ",
    "EARTH_MEAN_RADIUS",
    "LocalOrigin",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "enu_to_ned",
    "ned_to_enu",
    "enu_yaw_to_ned_yaw",
    "ned_yaw_to_enu_yaw",
    "haversine_distance",
    "initial_bearing",
    "destination_point",
    "wrap_pi",
    "wrap_2pi",
]

# --- WGS84 ellipsoid constants (defining parameters + derived) ---------------
WGS84_A: float = 6378137.0
"""Semi-major axis [m] (defining constant)."""

WGS84_F: float = 1.0 / 298.257223563
"""Flattening (defining constant)."""

WGS84_B: float = WGS84_A * (1.0 - WGS84_F)
"""Semi-minor axis [m]."""

WGS84_E_SQ: float = WGS84_F * (2.0 - WGS84_F)
"""First eccentricity squared."""

EARTH_MEAN_RADIUS: float = 6371008.8
"""IUGG mean radius [m], used only for spherical haversine helpers."""


def wrap_pi(angle: float) -> float:
    """Wrap an angle in radians to ``(-pi, pi]``."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def wrap_2pi(angle: float) -> float:
    """Wrap an angle in radians to ``[0, 2*pi)``."""
    wrapped = math.fmod(angle, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    return wrapped


# --- Geodetic <-> ECEF -------------------------------------------------------
def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates to Earth-Centered Earth-Fixed metres.

    Args:
        lat_deg: Geodetic latitude in degrees, positive north.
        lon_deg: Geodetic longitude in degrees, positive east.
        alt_m: Height above the WGS84 ellipsoid in metres.

    Returns:
        ``(x, y, z)`` in metres in the ECEF frame.

    Note:
        ``alt_m`` is **ellipsoidal** height, not height above mean sea level.
        GPS receivers report ellipsoidal height natively; anything labelled AMSL
        has already had a geoid model applied and differs by tens of metres in
        many parts of the world. Mixing the two is a classic reason a "50 m"
        survey altitude comes out wrong. This module never applies a geoid
        model -- feed it one kind of altitude consistently.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    # Radius of curvature in the prime vertical.
    n = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E_SQ) + alt_m) * sin_lat
    return (x, y, z)


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF metres back to WGS84 geodetic ``(lat_deg, lon_deg, alt_m)``.

    Uses Bowring's method with one refinement iteration, then a fixed-point
    polish. Accurate to well under a millimetre for any altitude a drone will
    ever see.
    """
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:
        # On the spin axis: latitude is +/-90, longitude is arbitrary.
        lat = math.copysign(math.pi / 2.0, z)
        alt = abs(z) - WGS84_B
        return (math.degrees(lat), math.degrees(lon), alt)

    # Bowring initial guess via the parametric latitude.
    ep_sq = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B)
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lat = math.atan2(
        z + ep_sq * WGS84_B * math.sin(theta) ** 3,
        p - WGS84_E_SQ * WGS84_A * math.cos(theta) ** 3,
    )
    # Fixed-point polish; converges in 2-3 rounds at drone altitudes.
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat_next = math.atan2(z, p * (1.0 - WGS84_E_SQ * n / (n + alt)))
        if abs(lat_next - lat) < 1e-14:
            lat = lat_next
            break
        lat = lat_next
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return (math.degrees(lat), math.degrees(lon), alt)


# --- ENU <-> NED -------------------------------------------------------------
def enu_to_ned(east: float, north: float, up: float) -> Tuple[float, float, float]:
    """Convert an ENU (ROS/REP-103) vector to NED (PX4/aviation).

    The mapping is ``(n, e, d) = (north, east, -up)``: swap the first two
    components, negate the third. It is an involution -- applying it twice gets
    you back where you started -- which is why :func:`ned_to_enu` looks
    identical. Both are provided so call sites read as documentation.
    """
    return (north, east, -up)


def ned_to_enu(north: float, east: float, down: float) -> Tuple[float, float, float]:
    """Convert a NED (PX4/aviation) vector to ENU (ROS/REP-103)."""
    return (east, north, -down)


def enu_yaw_to_ned_yaw(yaw_enu: float) -> float:
    """Convert an ENU yaw (CCW from East) to a NED heading (CW from North).

    ``yaw_ned = pi/2 - yaw_enu``, wrapped to ``(-pi, pi]``. Self-inverse.
    """
    return wrap_pi(math.pi / 2.0 - yaw_enu)


def ned_yaw_to_enu_yaw(yaw_ned: float) -> float:
    """Convert a NED heading (CW from North) to an ENU yaw (CCW from East)."""
    return wrap_pi(math.pi / 2.0 - yaw_ned)


# --- Spherical helpers -------------------------------------------------------
def haversine_distance(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> float:
    """Great-circle distance in metres between two points on a sphere.

    Uses the haversine formula on :data:`EARTH_MEAN_RADIUS`. Good to roughly
    0.3% against the ellipsoid, which is fine for "am I within 30 m of the
    waypoint" checks and wrong for survey-grade work -- use the ENU projection
    from :class:`LocalOrigin` for that.
    """
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    d_lat = lat2 - lat1
    d_lon = math.radians(lon2_deg - lon1_deg)
    a = math.sin(d_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
    return 2.0 * EARTH_MEAN_RADIUS * math.asin(math.sqrt(min(1.0, a)))


def initial_bearing(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> float:
    """Initial great-circle bearing from point 1 to point 2.

    Returns:
        Bearing in **degrees clockwise from true north**, in ``[0, 360)``.
        This is a compass heading (NED convention), not an ENU yaw. Convert
        with :func:`ned_yaw_to_enu_yaw` if you are feeding a ROS setpoint.
    """
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    d_lon = math.radians(lon2_deg - lon1_deg)
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    return math.degrees(wrap_2pi(math.atan2(y, x)))


def destination_point(
    lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float
) -> Tuple[float, float]:
    """Project a point along a great circle.

    Args:
        lat_deg: Start latitude in degrees.
        lon_deg: Start longitude in degrees.
        bearing_deg: Bearing clockwise from true north, degrees.
        distance_m: Distance to travel along the great circle, metres.

    Returns:
        ``(lat_deg, lon_deg)`` of the destination.
    """
    ang = distance_m / EARTH_MEAN_RADIUS
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brg = math.radians(bearing_deg)
    sin_lat2 = math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brg)
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(wrap_pi(lon2)))


# --- Local tangent plane -----------------------------------------------------
@dataclass(frozen=True)
class LocalOrigin:
    """A fixed local tangent-plane origin for ENU/NED projection.

    Anchor this once -- at the home/arming position, or at a surveyed pad -- and
    keep it for the whole flight. Re-anchoring mid-flight makes every stored
    local coordinate silently wrong, which shows up as the drone jumping when
    you reload a mission.

    Example:
        >>> origin = LocalOrigin(47.397742, 8.545594, 488.0)
        >>> e, n, u = origin.geodetic_to_enu(47.397742, 8.545594, 498.0)
        >>> round(e, 6), round(n, 6), round(u, 6)
        (0.0, 0.0, 10.0)
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(f"latitude_deg out of range: {self.latitude_deg}")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError(f"longitude_deg out of range: {self.longitude_deg}")

    # -- internals ----------------------------------------------------------
    @property
    def _ecef(self) -> Tuple[float, float, float]:
        return geodetic_to_ecef(self.latitude_deg, self.longitude_deg, self.altitude_m)

    @property
    def _trig(self) -> Tuple[float, float, float, float]:
        lat = math.radians(self.latitude_deg)
        lon = math.radians(self.longitude_deg)
        return (math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon))

    # -- ENU ----------------------------------------------------------------
    def geodetic_to_enu(
        self, lat_deg: float, lon_deg: float, alt_m: float
    ) -> Tuple[float, float, float]:
        """Project a geodetic point into local ENU metres relative to this origin."""
        x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        x0, y0, z0 = self._ecef
        dx, dy, dz = x - x0, y - y0, z - z0
        sin_lat, cos_lat, sin_lon, cos_lon = self._trig
        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        return (east, north, up)

    def enu_to_geodetic(
        self, east: float, north: float, up: float
    ) -> Tuple[float, float, float]:
        """Inverse of :meth:`geodetic_to_enu`."""
        sin_lat, cos_lat, sin_lon, cos_lon = self._trig
        dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
        dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
        dz = cos_lat * north + sin_lat * up
        x0, y0, z0 = self._ecef
        return ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)

    # -- NED ----------------------------------------------------------------
    def geodetic_to_ned(
        self, lat_deg: float, lon_deg: float, alt_m: float
    ) -> Tuple[float, float, float]:
        """Project a geodetic point into local NED metres (PX4 convention)."""
        east, north, up = self.geodetic_to_enu(lat_deg, lon_deg, alt_m)
        return enu_to_ned(east, north, up)

    def ned_to_geodetic(
        self, north: float, east: float, down: float
    ) -> Tuple[float, float, float]:
        """Inverse of :meth:`geodetic_to_ned`."""
        e, n, u = ned_to_enu(north, east, down)
        return self.enu_to_geodetic(e, n, u)

    # -- convenience --------------------------------------------------------
    def distance_to(self, lat_deg: float, lon_deg: float, alt_m: float) -> float:
        """3D straight-line distance in metres from the origin to a geodetic point."""
        east, north, up = self.geodetic_to_enu(lat_deg, lon_deg, alt_m)
        return math.sqrt(east * east + north * north + up * up)

    def ground_distance_to(self, lat_deg: float, lon_deg: float) -> float:
        """Horizontal (E/N only) distance in metres from the origin."""
        east, north, _ = self.geodetic_to_enu(lat_deg, lon_deg, self.altitude_m)
        return math.hypot(east, north)
