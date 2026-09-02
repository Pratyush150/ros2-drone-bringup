"""End-to-end tests over the core modules, with no ROS in the loop.

A tiny kinematic simulator flies the shipped example mission through the real
state machine while the real geofence watches. It is not a flight-dynamics
model -- it moves a point at the commanded speed -- but it exercises exactly
the interactions that break in the field: waypoint acceptance, mission
completion, geofence breach handling, and abort routing.
"""

import math

import pytest
import yaml

from drone_bringup.core.geofence import Geofence
from drone_bringup.core.mission import load_mission_file
from drone_bringup.core.state_machine import (
    AbortReason,
    MissionState,
    MissionStateMachine,
    VehicleSnapshot,
)


def healthy_snapshot(**overrides) -> VehicleSnapshot:
    """A vehicle that passes preflight."""
    base = dict(
        gps_fix_type=3,
        satellites=16,
        hdop=0.7,
        ekf_ok=True,
        battery_voltage=16.6,
        battery_percent=0.95,
        rc_connected=True,
        geofence_loaded=True,
        home_set=True,
        telemetry_age_s=0.02,
    )
    base.update(overrides)
    return VehicleSnapshot(**base)


class PointVehicle:
    """A point mass that flies straight at the commanded speed. Deterministic."""

    def __init__(self, east: float = 0.0, north: float = 0.0, up: float = 0.0) -> None:
        self.east = east
        self.north = north
        self.up = up
        self.vel = (0.0, 0.0, 0.0)

    def step_towards(self, target, speed: float, dt: float) -> float:
        """Move ``speed * dt`` towards ``target``; return the remaining distance."""
        de = target[0] - self.east
        dn = target[1] - self.north
        du = target[2] - self.up
        dist = math.sqrt(de * de + dn * dn + du * du)
        if dist < 1e-9:
            self.vel = (0.0, 0.0, 0.0)
            return 0.0
        step = min(dist, speed * dt)
        ux, uy, uz = de / dist, dn / dist, du / dist
        self.east += ux * step
        self.north += uy * step
        self.up += uz * step
        self.vel = (ux * speed, uy * speed, uz * speed)
        return dist - step


@pytest.fixture
def example_mission(config_dir):
    return load_mission_file(str(config_dir / "example_mission.yaml"))


@pytest.fixture
def example_fence(config_dir):
    with open(config_dir / "example_geofence.yaml", encoding="utf-8") as handle:
        return Geofence.from_dict(yaml.safe_load(handle))


class TestNominalFlight:
    """Fly the whole example mission and land."""

    def _fly(self, mission, fence, dt=0.5, max_steps=200000):
        waypoints = mission.expand()
        sm = MissionStateMachine()
        sm.begin_mission(len(waypoints))
        sm.start_preflight()
        sm.run_preflight(healthy_snapshot())
        sm.confirm_armed()

        vehicle = PointVehicle()
        breaches = 0
        steps = 0
        while sm.state is not MissionState.DISARMED and steps < max_steps:
            steps += 1
            if sm.state is MissionState.TAKEOFF:
                target = mission.waypoint_to_enu(waypoints[0])
                if vehicle.step_towards(target, 3.0, dt) <= 1.0:
                    sm.confirm_takeoff_complete()
            elif sm.state is MissionState.MISSION:
                waypoint = waypoints[sm.waypoint_index]
                target = mission.waypoint_to_enu(waypoint)
                remaining = vehicle.step_towards(target, waypoint.speed, dt)
                if remaining <= waypoint.acceptance_radius:
                    if not sm.advance_waypoint():
                        sm.finish_mission()
            elif sm.state is MissionState.RTL:
                # A real RTL climbs to the RTL altitude first, then transits,
                # then descends. Cutting the corner and flying a straight line
                # home is what puts a vehicle through the keepout it was told
                # to avoid.
                rtl_alt = mission.rtl_altitude
                if vehicle.up < rtl_alt - 0.5:
                    vehicle.step_towards(
                        (vehicle.east, vehicle.north, rtl_alt), 4.0, dt
                    )
                elif (
                    vehicle.step_towards((0.0, 0.0, rtl_alt), 8.0, dt) <= 1.0
                ):
                    sm.begin_landing()
            elif sm.state is MissionState.LANDING:
                if vehicle.step_towards((0.0, 0.0, 0.0), 1.5, dt) <= 0.2:
                    sm.confirm_disarmed()

            status = fence.check_local(
                vehicle.east, vehicle.north, vehicle.up, *vehicle.vel
            )
            if status.breached:
                breaches += 1
        return sm, vehicle, breaches, steps

    def test_mission_reaches_disarmed(self, example_mission, example_fence):
        sm, _, _, steps = self._fly(example_mission, example_fence)
        assert sm.state is MissionState.DISARMED
        assert steps < 200000

    def test_every_waypoint_is_visited(self, example_mission, example_fence):
        sm, _, _, _ = self._fly(example_mission, example_fence)
        assert sm.waypoint_index == len(example_mission.expand())
        assert sm.mission_complete

    def test_geofence_is_never_breached(self, example_mission, example_fence):
        _, _, breaches, _ = self._fly(example_mission, example_fence)
        assert breaches == 0

    def test_vehicle_ends_at_the_origin(self, example_mission, example_fence):
        _, vehicle, _, _ = self._fly(example_mission, example_fence)
        assert math.hypot(vehicle.east, vehicle.north) < 1.0
        assert vehicle.up < 0.3

    def test_no_aborts_were_raised(self, example_mission, example_fence):
        sm, _, _, _ = self._fly(example_mission, example_fence)
        assert sm.abort_reason is None

    def test_state_sequence_is_the_nominal_one(self, example_mission, example_fence):
        sm, _, _, _ = self._fly(example_mission, example_fence)
        assert [to for _, to in sm.history_states()] == [
            "PREFLIGHT",
            "ARMING",
            "TAKEOFF",
            "MISSION",
            "RTL",
            "LANDING",
            "DISARMED",
        ]


class TestGeofenceAbort:
    """A vehicle heading out of the fence must be caught before it leaves."""

    def test_predicted_breach_triggers_an_abort_before_the_boundary(
        self, example_mission, example_fence
    ):
        sm = MissionStateMachine()
        sm.begin_mission(len(example_mission.expand()))
        sm.start_preflight()
        sm.run_preflight(healthy_snapshot())
        sm.confirm_armed()
        sm.confirm_takeoff_complete()

        # Start well inside and fly east at 15 m/s.
        vehicle = PointVehicle(east=100.0, north=100.0, up=40.0)
        aborted_at = None
        for _ in range(2000):
            vehicle.east += 15.0 * 0.1
            status = example_fence.check_local(
                vehicle.east, vehicle.north, vehicle.up, 15.0, 0.0, 0.0
            )
            assert not status.breached, "abort should have fired before the boundary"
            breach = status.predicted_breach
            if breach is not None and breach.time_to_breach_s < 4.0:
                # distance_m is the distance to THAT boundary. margin_m is the
                # tightest margin across every constraint, which here is the
                # altitude floor, so it is the wrong number to assert on.
                aborted_at = breach.distance_m
                sm.abort(AbortReason.GEOFENCE_BREACH, str(breach))
                break
        assert aborted_at is not None
        # 15 m/s x 4 s = 60 m of warning.
        assert aborted_at == pytest.approx(60.0, abs=2.0)
        assert sm.state is MissionState.RTL

    def test_faster_vehicle_gets_more_distance_warning(self, example_fence):
        # Same time horizon, higher speed -> the warning fires further out.
        def margin_at_trigger(speed):
            east = 100.0
            while True:
                east += speed * 0.05
                status = example_fence.check_local(east, 100.0, 40.0, speed, 0.0, 0.0)
                breach = status.predicted_breach
                if breach is not None and breach.time_to_breach_s < 4.0:
                    return breach.distance_m

        assert margin_at_trigger(20.0) > margin_at_trigger(5.0)

    def test_critical_battery_mid_mission_lands_immediately(
        self, example_mission, example_fence
    ):
        sm = MissionStateMachine()
        sm.begin_mission(len(example_mission.expand()))
        sm.start_preflight()
        sm.run_preflight(healthy_snapshot())
        sm.confirm_armed()
        sm.confirm_takeoff_complete()
        assert sm.abort(AbortReason.CRITICAL_BATTERY, "3.1 V/cell") is MissionState.LANDING
        assert sm.confirm_disarmed() is MissionState.DISARMED


class TestPreflightBlocksFlight:
    """A failing preflight must stop the sortie before anything spins."""

    def test_bad_gps_never_reaches_arming(self):
        sm = MissionStateMachine()
        sm.start_preflight()
        status = sm.run_preflight(healthy_snapshot(gps_fix_type=0, satellites=2))
        assert not status.passed
        assert sm.state is MissionState.FAULT
        assert MissionState.ARMING not in [r.to_state for r in sm.history]

    def test_missing_geofence_blocks_arming(self):
        sm = MissionStateMachine()
        sm.start_preflight()
        sm.run_preflight(healthy_snapshot(geofence_loaded=False))
        assert sm.state is MissionState.FAULT

    def test_operator_can_reset_and_retry_after_a_fix(self):
        sm = MissionStateMachine()
        sm.start_preflight()
        sm.run_preflight(healthy_snapshot(ekf_ok=False))
        assert sm.state is MissionState.FAULT
        sm.reset()
        sm.start_preflight()
        assert sm.run_preflight(healthy_snapshot()).passed
        assert sm.state is MissionState.ARMING


class TestMissionGeofenceConsistency:
    """The shipped mission and geofence must agree with each other."""

    def test_survey_altitude_is_under_the_ceiling(self, example_mission, example_fence):
        ceiling = example_fence.max_altitude_m
        assert ceiling is not None
        assert all(w.altitude < ceiling for w in example_mission.expand())

    def test_mission_origin_matches_the_fence_origin(self, example_mission, example_fence):
        assert example_mission.origin.latitude_deg == pytest.approx(
            example_fence.origin.latitude_deg
        )
        assert example_mission.origin.longitude_deg == pytest.approx(
            example_fence.origin.longitude_deg
        )

    def test_orbit_clears_the_exclusion_zone(self, example_mission, example_fence):
        keepout = example_fence.exclusion_zones[0]
        orbit_points = [
            w for w in example_mission.expand() if w.label.startswith("orbit")
        ]
        assert orbit_points
        for waypoint in orbit_points:
            east, north, _ = example_mission.waypoint_to_enu(waypoint)
            assert not keepout.polygon.contains(east, north)

    def test_mission_duration_is_plausible_for_one_battery(self, example_mission):
        # A lower bound, not a promise -- it ignores wind and turn losses.
        minutes = example_mission.estimated_duration_s() / 60.0
        assert 1.0 < minutes < 25.0
