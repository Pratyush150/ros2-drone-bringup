"""Polygon + altitude geofence with breach prediction.

What this gives you
-------------------
* Inclusion zones (must stay inside) and exclusion zones (must stay out of),
  each an arbitrary simple polygon.
* An altitude ceiling and floor, checked against the local tangent plane.
* Signed distance to the nearest boundary, positive when safe.
* **Time to breach**: given the current velocity, when does the vehicle cross a
  boundary if nothing changes. This is the number you actually gate on. A
  binary "am I inside" check fires when it is already too late -- a 15 m/s
  multirotor needs several seconds and tens of metres to stop, so you want to
  trigger a hold or RTL on predicted breach, not on actual breach.

All geometry runs in local ENU metres against a :class:`~drone_bringup.core.
geodesy.LocalOrigin`, so it is exact enough for any site you would fly and does
not care about longitude convergence. Geodetic wrappers are provided for
convenience.

Pure Python, no ROS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from .geodesy import LocalOrigin

__all__ = [
    "GeofenceError",
    "ZoneKind",
    "BreachKind",
    "Polygon2D",
    "GeofenceZone",
    "Geofence",
    "GeofenceStatus",
    "BreachPrediction",
    "point_in_polygon",
    "polygon_area",
    "polygon_centroid",
    "zones_from_local",
]

_EPS = 1e-9


class GeofenceError(ValueError):
    """Raised when a geofence definition is structurally invalid."""


class ZoneKind(str, Enum):
    """Whether a polygon must contain the vehicle or must not."""

    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class BreachKind(str, Enum):
    """What kind of boundary a predicted or actual breach involves."""

    INCLUSION_EXIT = "inclusion_exit"
    EXCLUSION_ENTRY = "exclusion_entry"
    ALTITUDE_CEILING = "altitude_ceiling"
    ALTITUDE_FLOOR = "altitude_floor"


# --- free-standing polygon helpers (metres, 2D) ------------------------------
def point_in_polygon(x: float, y: float, vertices: Sequence[Tuple[float, float]]) -> bool:
    """Even-odd ray-casting point-in-polygon test.

    The polygon is treated as implicitly closed. Points exactly on an edge are
    not guaranteed either way -- that is inherent to the test and irrelevant at
    metre scale, but do not build an equality assertion on it.
    """
    inside = False
    n = len(vertices)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x_cross > x:
                inside = not inside
        j = i
    return inside


def polygon_area(vertices: Sequence[Tuple[float, float]]) -> float:
    """Signed shoelace area. Positive for counter-clockwise winding."""
    total = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def polygon_centroid(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Area centroid of a simple polygon; falls back to the vertex mean if degenerate."""
    area = polygon_area(vertices)
    if abs(area) < _EPS:
        n = float(len(vertices))
        return (sum(v[0] for v in vertices) / n, sum(v[1] for v in vertices) / n)
    cx = cy = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6.0 * area), cy / (6.0 * area))


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Shortest distance from point P to segment AB."""
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < _EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _ray_segment_time(
    px: float, py: float, vx: float, vy: float, ax: float, ay: float, bx: float, by: float
) -> Optional[float]:
    """Earliest ``t >= 0`` at which ``P + t*V`` crosses segment AB, or ``None``."""
    ex, ey = bx - ax, by - ay
    denom = vx * ey - vy * ex
    if abs(denom) < _EPS:
        return None  # parallel (or stationary): never crosses, or already on it
    wx, wy = ax - px, ay - py
    t = (wx * ey - wy * ex) / denom
    s = (wx * vy - wy * vx) / denom
    if t < 0.0 or not (-_EPS <= s <= 1.0 + _EPS):
        return None
    return t


@dataclass(frozen=True)
class Polygon2D:
    """A simple polygon in local ENU metres, ``(east, north)`` per vertex."""

    vertices: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise GeofenceError(
                f"polygon needs at least 3 vertices, got {len(self.vertices)}"
            )

    def contains(self, x: float, y: float) -> bool:
        """True if ``(x, y)`` lies inside the polygon."""
        return point_in_polygon(x, y, self.vertices)

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Unsigned distance in metres from ``(x, y)`` to the nearest edge."""
        n = len(self.vertices)
        return min(
            _point_segment_distance(
                x, y, *self.vertices[i], *self.vertices[(i + 1) % n]
            )
            for i in range(n)
        )

    def signed_distance(self, x: float, y: float) -> float:
        """Distance to the boundary, **positive inside** and negative outside."""
        d = self.distance_to_boundary(x, y)
        return d if self.contains(x, y) else -d

    def time_to_cross(
        self, x: float, y: float, vx: float, vy: float
    ) -> Optional[float]:
        """Earliest time in seconds at which a constant-velocity ray crosses an edge.

        Returns ``None`` if the velocity is zero or the ray never reaches the
        boundary. Works the same whether the point starts inside (time to exit)
        or outside (time to enter).
        """
        if math.hypot(vx, vy) < _EPS:
            return None
        n = len(self.vertices)
        best: Optional[float] = None
        for i in range(n):
            t = _ray_segment_time(
                x, y, vx, vy, *self.vertices[i], *self.vertices[(i + 1) % n]
            )
            if t is not None and (best is None or t < best):
                best = t
        return best

    @property
    def area(self) -> float:
        """Absolute area in square metres."""
        return abs(polygon_area(self.vertices))

    @property
    def centroid(self) -> Tuple[float, float]:
        """Area centroid in local metres."""
        return polygon_centroid(self.vertices)


@dataclass(frozen=True)
class GeofenceZone:
    """A named polygon zone with an optional per-zone altitude band."""

    name: str
    kind: ZoneKind
    polygon: Polygon2D
    min_alt_m: Optional[float] = None
    max_alt_m: Optional[float] = None

    def altitude_applies(self, up_m: float) -> bool:
        """True if ``up_m`` is inside this zone's vertical band (or it has none)."""
        if self.min_alt_m is not None and up_m < self.min_alt_m:
            return False
        if self.max_alt_m is not None and up_m > self.max_alt_m:
            return False
        return True


@dataclass(frozen=True)
class BreachPrediction:
    """A predicted boundary crossing."""

    kind: BreachKind
    zone_name: str
    time_to_breach_s: float
    distance_m: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.kind.value} on '{self.zone_name}' in "
            f"{self.time_to_breach_s:.1f} s ({self.distance_m:.1f} m)"
        )


@dataclass(frozen=True)
class GeofenceStatus:
    """Result of evaluating a position (and optionally a velocity) against a fence."""

    inside: bool
    violations: Tuple[str, ...] = ()
    margin_m: float = math.inf
    predicted_breach: Optional[BreachPrediction] = None

    @property
    def breached(self) -> bool:
        """True if the vehicle is *currently* outside the allowed region."""
        return not self.inside


@dataclass
class Geofence:
    """A complete fence: zones plus a global altitude band, anchored at an origin.

    All altitudes are **metres Up** in the local ENU frame, i.e. relative to the
    origin, not AMSL. Fixing that convention here means the monitor node never
    has to reason about geoid offsets.
    """

    origin: LocalOrigin
    zones: List[GeofenceZone] = field(default_factory=list)
    max_altitude_m: Optional[float] = None
    min_altitude_m: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            self.max_altitude_m is not None
            and self.min_altitude_m is not None
            and self.max_altitude_m <= self.min_altitude_m
        ):
            raise GeofenceError(
                f"max_altitude_m ({self.max_altitude_m}) must exceed "
                f"min_altitude_m ({self.min_altitude_m})"
            )
        names = [z.name for z in self.zones]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise GeofenceError(f"duplicate zone names: {sorted(dupes)}")

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "Geofence":
        """Build a fence from a plain dict (as loaded from the geofence YAML).

        Expected shape::

            origin: {latitude: 47.3977, longitude: 8.5456, altitude: 488.0}
            max_altitude_m: 120.0
            min_altitude_m: -5.0
            zones:
              - name: survey_area
                kind: inclusion
                vertices: [[lat, lon], ...]

        Raises:
            GeofenceError: With a message naming the offending key.
        """
        if not isinstance(data, dict):
            raise GeofenceError(f"geofence root must be a mapping, got {type(data).__name__}")
        origin_raw = data.get("origin")
        if not isinstance(origin_raw, dict):
            raise GeofenceError("geofence: missing required 'origin' mapping")
        for key in ("latitude", "longitude"):
            if key not in origin_raw:
                raise GeofenceError(f"geofence.origin: missing required key '{key}'")
        origin = LocalOrigin(
            float(origin_raw["latitude"]),
            float(origin_raw["longitude"]),
            float(origin_raw.get("altitude", 0.0)),
        )
        zones: List[GeofenceZone] = []
        raw_zones = data.get("zones", [])
        if not isinstance(raw_zones, list):
            raise GeofenceError("geofence.zones must be a list")
        for i, raw in enumerate(raw_zones):
            if not isinstance(raw, dict):
                raise GeofenceError(f"geofence.zones[{i}] must be a mapping")
            name = str(raw.get("name", f"zone_{i}"))
            kind_raw = str(raw.get("kind", "inclusion")).lower()
            try:
                kind = ZoneKind(kind_raw)
            except ValueError:
                raise GeofenceError(
                    f"geofence.zones[{i}] ('{name}'): kind must be one of "
                    f"{[k.value for k in ZoneKind]}, got '{kind_raw}'"
                ) from None
            verts_raw = raw.get("vertices")
            if not isinstance(verts_raw, list) or len(verts_raw) < 3:
                raise GeofenceError(
                    f"geofence.zones[{i}] ('{name}'): 'vertices' must be a list of "
                    f"at least 3 [lat, lon] pairs"
                )
            local: List[Tuple[float, float]] = []
            for j, v in enumerate(verts_raw):
                if not isinstance(v, (list, tuple)) or len(v) != 2:
                    raise GeofenceError(
                        f"geofence.zones[{i}].vertices[{j}] must be a [lat, lon] pair"
                    )
                e, n, _ = origin.geodetic_to_enu(float(v[0]), float(v[1]), origin.altitude_m)
                local.append((e, n))
            zones.append(
                GeofenceZone(
                    name=name,
                    kind=kind,
                    polygon=Polygon2D(tuple(local)),
                    min_alt_m=_opt_float(raw.get("min_altitude_m")),
                    max_alt_m=_opt_float(raw.get("max_altitude_m")),
                )
            )
        return cls(
            origin=origin,
            zones=zones,
            max_altitude_m=_opt_float(data.get("max_altitude_m")),
            min_altitude_m=_opt_float(data.get("min_altitude_m")),
        )

    # -- evaluation ---------------------------------------------------------
    def check_local(
        self,
        east: float,
        north: float,
        up: float,
        vel_east: float = 0.0,
        vel_north: float = 0.0,
        vel_up: float = 0.0,
        horizon_s: float = 30.0,
    ) -> GeofenceStatus:
        """Evaluate a local-ENU position and velocity against the fence.

        Args:
            east: Local ENU east position, metres.
            north: Local ENU north position, metres.
            up: Local ENU up position, metres.
            vel_east: Velocity east, m/s. Zero disables horizontal prediction.
            vel_north: Velocity north, m/s.
            vel_up: Velocity up, m/s.
            horizon_s: Only report predicted breaches inside this many seconds.

        Returns:
            A :class:`GeofenceStatus`. ``margin_m`` is the smallest safety
            margin across every active constraint: horizontal distance to the
            nearest relevant polygon boundary and vertical distance to the
            nearest altitude limit, whichever is tighter.
        """
        violations: List[str] = []
        margin = math.inf
        predictions: List[BreachPrediction] = []

        # Altitude band.
        if self.max_altitude_m is not None:
            gap = self.max_altitude_m - up
            margin = min(margin, gap)
            if gap < 0.0:
                violations.append(
                    f"altitude {up:.1f} m exceeds ceiling {self.max_altitude_m:.1f} m"
                )
            elif vel_up > _EPS:
                predictions.append(
                    BreachPrediction(
                        BreachKind.ALTITUDE_CEILING, "ceiling", gap / vel_up, gap
                    )
                )
        if self.min_altitude_m is not None:
            gap = up - self.min_altitude_m
            margin = min(margin, gap)
            if gap < 0.0:
                violations.append(
                    f"altitude {up:.1f} m below floor {self.min_altitude_m:.1f} m"
                )
            elif vel_up < -_EPS:
                predictions.append(
                    BreachPrediction(
                        BreachKind.ALTITUDE_FLOOR, "floor", gap / -vel_up, gap
                    )
                )

        # Polygon zones.
        for zone in self.zones:
            if not zone.altitude_applies(up):
                continue
            inside_poly = zone.polygon.contains(east, north)
            dist = zone.polygon.distance_to_boundary(east, north)
            if zone.kind is ZoneKind.INCLUSION:
                margin = min(margin, dist if inside_poly else -dist)
                if not inside_poly:
                    violations.append(
                        f"outside inclusion zone '{zone.name}' by {dist:.1f} m"
                    )
                else:
                    t = zone.polygon.time_to_cross(east, north, vel_east, vel_north)
                    if t is not None:
                        predictions.append(
                            BreachPrediction(
                                BreachKind.INCLUSION_EXIT, zone.name, t, dist
                            )
                        )
            else:  # EXCLUSION
                margin = min(margin, -dist if inside_poly else dist)
                if inside_poly:
                    violations.append(
                        f"inside exclusion zone '{zone.name}' by {dist:.1f} m"
                    )
                else:
                    t = zone.polygon.time_to_cross(east, north, vel_east, vel_north)
                    if t is not None:
                        predictions.append(
                            BreachPrediction(
                                BreachKind.EXCLUSION_ENTRY, zone.name, t, dist
                            )
                        )

        soonest = None
        candidates = [p for p in predictions if p.time_to_breach_s <= horizon_s]
        if candidates:
            soonest = min(candidates, key=lambda p: p.time_to_breach_s)

        return GeofenceStatus(
            inside=not violations,
            violations=tuple(violations),
            margin_m=margin,
            predicted_breach=soonest,
        )

    def check_geodetic(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        vel_east: float = 0.0,
        vel_north: float = 0.0,
        vel_up: float = 0.0,
        horizon_s: float = 30.0,
    ) -> GeofenceStatus:
        """Same as :meth:`check_local` but takes a WGS84 position."""
        e, n, u = self.origin.geodetic_to_enu(lat_deg, lon_deg, alt_m)
        return self.check_local(e, n, u, vel_east, vel_north, vel_up, horizon_s)

    def time_to_breach(
        self,
        east: float,
        north: float,
        up: float,
        vel_east: float,
        vel_north: float,
        vel_up: float,
    ) -> Optional[float]:
        """Convenience: seconds until the first predicted breach, or ``None``."""
        status = self.check_local(
            east, north, up, vel_east, vel_north, vel_up, horizon_s=math.inf
        )
        if status.predicted_breach is None:
            return None
        return status.predicted_breach.time_to_breach_s

    @property
    def inclusion_zones(self) -> List[GeofenceZone]:
        """Every inclusion zone, in declaration order."""
        return [z for z in self.zones if z.kind is ZoneKind.INCLUSION]

    @property
    def exclusion_zones(self) -> List[GeofenceZone]:
        """Every exclusion zone, in declaration order."""
        return [z for z in self.zones if z.kind is ZoneKind.EXCLUSION]


def _opt_float(value: object) -> Optional[float]:
    """Coerce to float, passing ``None`` through."""
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


def zones_from_local(
    origin: LocalOrigin, specs: Iterable[Tuple[str, ZoneKind, Sequence[Tuple[float, float]]]]
) -> List[GeofenceZone]:
    """Helper to build zones straight from local ENU vertex lists (mostly for tests)."""
    return [
        GeofenceZone(name=name, kind=kind, polygon=Polygon2D(tuple(verts)))
        for name, kind, verts in specs
    ]
