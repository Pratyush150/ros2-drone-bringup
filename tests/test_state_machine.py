"""Tests for the mission executor state machine.

Covers the nominal path, every abort route, and -- most importantly -- that
illegal transitions are *rejected* rather than silently applied. A state machine
that accepts anything is worse than no state machine, because it looks like
safety.
"""

import itertools

import pytest

from drone_bringup.core.state_machine import (
    AIRBORNE_STATES,
    TRANSITIONS,
    AbortReason,
    IllegalTransition,
    MissionState,
    MissionStateMachine,
    PreflightLimits,
    VehicleSnapshot,
    check_preflight,
    reachable_states,
)

ALL_STATES = list(MissionState)


def good_snapshot(**overrides) -> VehicleSnapshot:
    """A snapshot that passes every default preflight gate."""
    base = dict(
        gps_fix_type=3,
        satellites=14,
        hdop=0.8,
        ekf_ok=True,
        battery_voltage=16.4,
        battery_percent=0.92,
        rc_connected=True,
        geofence_loaded=True,
        home_set=True,
        armed=False,
        telemetry_age_s=0.05,
        altitude_m=0.0,
    )
    base.update(overrides)
    return VehicleSnapshot(**base)


def fly_to(state: MissionState) -> MissionStateMachine:
    """Drive a fresh machine along the nominal path up to ``state``."""
    sm = MissionStateMachine()
    if state is MissionState.IDLE:
        return sm
    sm.start_preflight()
    if state is MissionState.PREFLIGHT:
        return sm
    sm.run_preflight(good_snapshot())
    if state is MissionState.ARMING:
        return sm
    sm.confirm_armed()
    if state is MissionState.TAKEOFF:
        return sm
    sm.confirm_takeoff_complete()
    if state is MissionState.MISSION:
        return sm
    sm.finish_mission()
    if state is MissionState.RTL:
        return sm
    sm.begin_landing()
    if state is MissionState.LANDING:
        return sm
    sm.confirm_disarmed()
    if state is MissionState.DISARMED:
        return sm
    raise ValueError(f"no nominal path to {state}")


class TestTransitionTable:
    """Structural properties of the table itself."""

    def test_every_state_has_an_entry(self):
        assert set(TRANSITIONS) == set(ALL_STATES)

    def test_no_state_transitions_to_itself(self):
        for state, targets in TRANSITIONS.items():
            assert state not in targets

    def test_every_state_is_reachable_from_idle(self):
        assert reachable_states(MissionState.IDLE) == frozenset(ALL_STATES)

    def test_every_state_can_reach_a_terminal_state(self):
        for state in ALL_STATES:
            reachable = reachable_states(state)
            assert reachable & {MissionState.DISARMED, MissionState.FAULT}

    def test_airborne_states_are_the_ones_after_takeoff(self):
        assert AIRBORNE_STATES == frozenset(
            {
                MissionState.TAKEOFF,
                MissionState.MISSION,
                MissionState.RTL,
                MissionState.LANDING,
                MissionState.ABORT,
            }
        )

    def test_every_airborne_state_can_abort_or_is_already_recovering(self):
        for state in AIRBORNE_STATES:
            if state is MissionState.ABORT:
                continue
            assert MissionState.ABORT in TRANSITIONS[state]

    def test_terminal_states_only_lead_back_to_idle(self):
        assert TRANSITIONS[MissionState.DISARMED] == frozenset({MissionState.IDLE})
        assert TRANSITIONS[MissionState.FAULT] == frozenset({MissionState.IDLE})


class TestNominalPath:
    """IDLE -> ... -> DISARMED."""

    def test_starts_in_idle(self):
        assert MissionStateMachine().state is MissionState.IDLE

    def test_full_nominal_sequence(self):
        sm = MissionStateMachine()
        sm.begin_mission(3)
        assert sm.start_preflight() is MissionState.PREFLIGHT
        assert sm.run_preflight(good_snapshot()).passed
        assert sm.state is MissionState.ARMING
        assert sm.confirm_armed() is MissionState.TAKEOFF
        assert sm.confirm_takeoff_complete() is MissionState.MISSION
        assert sm.finish_mission() is MissionState.RTL
        assert sm.begin_landing() is MissionState.LANDING
        assert sm.confirm_disarmed() is MissionState.DISARMED
        assert sm.is_terminal

    def test_history_records_every_edge(self):
        sm = fly_to(MissionState.DISARMED)
        assert sm.history_states() == [
            ("IDLE", "PREFLIGHT"),
            ("PREFLIGHT", "ARMING"),
            ("ARMING", "TAKEOFF"),
            ("TAKEOFF", "MISSION"),
            ("MISSION", "RTL"),
            ("RTL", "LANDING"),
            ("LANDING", "DISARMED"),
        ]

    def test_transition_callback_fires(self):
        seen = []
        sm = MissionStateMachine(on_transition=seen.append)
        sm.start_preflight(now=1.5)
        assert len(seen) == 1
        assert seen[0].to_state is MissionState.PREFLIGHT
        assert seen[0].timestamp == pytest.approx(1.5)

    def test_reset_returns_to_idle(self):
        sm = fly_to(MissionState.DISARMED)
        assert sm.reset() is MissionState.IDLE
        assert sm.abort_reason is None
        assert sm.waypoint_count == 0

    def test_airborne_flag(self):
        assert not fly_to(MissionState.PREFLIGHT).is_airborne
        assert not fly_to(MissionState.ARMING).is_airborne
        assert fly_to(MissionState.TAKEOFF).is_airborne
        assert fly_to(MissionState.MISSION).is_airborne
        assert not fly_to(MissionState.DISARMED).is_airborne

    def test_snapshot_is_serialisable(self):
        sm = fly_to(MissionState.MISSION)
        sm.begin_mission(5)
        snap = sm.snapshot()
        assert snap["state"] == "MISSION"
        assert snap["waypoint_count"] == 5
        assert snap["airborne"] is True


class TestIllegalTransitions:
    """The important half: what the machine refuses to do."""

    def test_cannot_skip_from_idle_to_mission(self):
        sm = MissionStateMachine()
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.MISSION)

    def test_cannot_arm_without_preflight(self):
        sm = MissionStateMachine()
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.ARMING)

    def test_cannot_take_off_from_preflight(self):
        sm = fly_to(MissionState.PREFLIGHT)
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.TAKEOFF)

    def test_cannot_go_straight_from_takeoff_to_disarmed(self):
        sm = fly_to(MissionState.TAKEOFF)
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.DISARMED)

    def test_cannot_go_back_from_mission_to_takeoff(self):
        sm = fly_to(MissionState.MISSION)
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.TAKEOFF)

    def test_cannot_resume_from_disarmed_to_mission(self):
        sm = fly_to(MissionState.DISARMED)
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.MISSION)

    def test_cannot_recover_from_fault_except_by_reset(self):
        for target in ALL_STATES:
            if target is MissionState.IDLE:
                continue
            with pytest.raises(IllegalTransition):
                MissionStateMachine(state=MissionState.FAULT).request(target)

    def test_rejected_transition_leaves_the_state_untouched(self):
        sm = fly_to(MissionState.MISSION)
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.ARMING)
        assert sm.state is MissionState.MISSION

    def test_rejected_transition_is_not_recorded_in_history(self):
        sm = fly_to(MissionState.MISSION)
        before = len(sm.history)
        with pytest.raises(IllegalTransition):
            sm.request(MissionState.ARMING)
        assert len(sm.history) == before

    def test_error_message_lists_the_legal_targets(self):
        sm = fly_to(MissionState.TAKEOFF)
        with pytest.raises(IllegalTransition) as exc:
            sm.request(MissionState.DISARMED)
        message = str(exc.value)
        assert "TAKEOFF -> DISARMED" in message
        assert "MISSION" in message and "ABORT" in message

    def test_exception_carries_both_states(self):
        sm = fly_to(MissionState.TAKEOFF)
        with pytest.raises(IllegalTransition) as exc:
            sm.request(MissionState.DISARMED)
        assert exc.value.current is MissionState.TAKEOFF
        assert exc.value.requested is MissionState.DISARMED

    def test_try_request_returns_false_instead_of_raising(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.try_request(MissionState.ARMING) is False
        assert sm.state is MissionState.MISSION

    def test_try_request_returns_true_on_success(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.try_request(MissionState.RTL) is True

    def test_can_matches_the_table(self):
        for state in ALL_STATES:
            sm = MissionStateMachine(state=state)
            for target in ALL_STATES:
                assert sm.can(target) == (target in TRANSITIONS[state])

    @pytest.mark.parametrize(
        "state,target",
        [
            (s, t)
            for s, t in itertools.product(ALL_STATES, ALL_STATES)
            if t not in TRANSITIONS[s]
        ],
    )
    def test_every_undeclared_edge_is_rejected(self, state, target):
        """Exhaustive: every (from, to) pair not in the table must raise."""
        sm = MissionStateMachine(state=state)
        with pytest.raises(IllegalTransition):
            sm.request(target)
        assert sm.state is state

    @pytest.mark.parametrize(
        "state,target",
        [(s, t) for s, targets in TRANSITIONS.items() for t in targets],
    )
    def test_every_declared_edge_is_accepted(self, state, target):
        """Exhaustive: every declared edge must actually work."""
        sm = MissionStateMachine(state=state)
        assert sm.request(target) is target


class TestPreflightGate:
    """check_preflight reports every failure, not just the first."""

    def test_good_snapshot_passes(self):
        status = check_preflight(good_snapshot())
        assert status.passed
        assert status.failures == ()

    def test_summary_text_for_a_pass(self):
        assert check_preflight(good_snapshot()).summary() == "preflight OK"

    def test_no_gps_fix_fails(self):
        status = check_preflight(good_snapshot(gps_fix_type=1))
        assert not status.passed
        assert any("GPS fix type" in f for f in status.failures)

    def test_too_few_satellites_fails(self):
        status = check_preflight(good_snapshot(satellites=4))
        assert any("satellites" in f for f in status.failures)

    def test_high_hdop_fails(self):
        status = check_preflight(good_snapshot(hdop=5.0))
        assert any("HDOP" in f for f in status.failures)

    def test_ekf_not_ready_fails(self):
        status = check_preflight(good_snapshot(ekf_ok=False))
        assert any("EKF" in f for f in status.failures)

    def test_low_voltage_fails(self):
        status = check_preflight(good_snapshot(battery_voltage=13.2))
        assert any("battery" in f and "V" in f for f in status.failures)

    def test_low_state_of_charge_fails(self):
        status = check_preflight(good_snapshot(battery_percent=0.20))
        assert any("%" in f for f in status.failures)

    def test_no_rc_fails_by_default(self):
        status = check_preflight(good_snapshot(rc_connected=False))
        assert any("RC link" in f for f in status.failures)

    def test_no_rc_can_be_allowed_for_sitl(self):
        limits = PreflightLimits(require_rc=False)
        assert check_preflight(good_snapshot(rc_connected=False), limits).passed

    def test_missing_geofence_fails(self):
        status = check_preflight(good_snapshot(geofence_loaded=False))
        assert any("geofence" in f for f in status.failures)

    def test_missing_home_fails(self):
        status = check_preflight(good_snapshot(home_set=False))
        assert any("home position" in f for f in status.failures)

    def test_stale_telemetry_fails(self):
        status = check_preflight(good_snapshot(telemetry_age_s=3.0))
        assert any("stale" in f for f in status.failures)

    def test_all_failures_are_reported_together(self):
        status = check_preflight(
            good_snapshot(
                gps_fix_type=0,
                satellites=2,
                ekf_ok=False,
                battery_percent=0.1,
                rc_connected=False,
            )
        )
        assert len(status.failures) >= 5

    def test_already_armed_is_a_warning_not_a_failure(self):
        status = check_preflight(good_snapshot(armed=True))
        assert status.passed
        assert any("already armed" in w for w in status.warnings)

    def test_summary_mentions_warnings_on_a_pass(self):
        summary = check_preflight(good_snapshot(armed=True)).summary()
        assert "with warnings" in summary

    def test_summary_lists_failures(self):
        summary = check_preflight(good_snapshot(ekf_ok=False)).summary()
        assert summary.startswith("preflight FAILED")

    def test_custom_limits_are_honoured(self):
        limits = PreflightLimits(min_battery_percent=0.95)
        assert not check_preflight(good_snapshot(battery_percent=0.92), limits).passed


class TestPreflightIntegration:
    """run_preflight wires the gate into the machine."""

    def test_pass_advances_to_arming(self):
        sm = fly_to(MissionState.PREFLIGHT)
        sm.run_preflight(good_snapshot())
        assert sm.state is MissionState.ARMING

    def test_failure_routes_to_fault_on_the_ground(self):
        sm = fly_to(MissionState.PREFLIGHT)
        status = sm.run_preflight(good_snapshot(ekf_ok=False))
        assert not status.passed
        assert sm.state is MissionState.FAULT
        assert sm.abort_reason is AbortReason.PREFLIGHT_FAILED

    def test_last_preflight_is_stored(self):
        sm = fly_to(MissionState.PREFLIGHT)
        sm.run_preflight(good_snapshot())
        assert sm.last_preflight is not None
        assert sm.last_preflight.passed

    def test_run_preflight_from_the_wrong_state_raises(self):
        sm = fly_to(MissionState.MISSION)
        with pytest.raises(IllegalTransition):
            sm.run_preflight(good_snapshot())

    def test_arming_rejection_goes_back_to_preflight(self):
        sm = fly_to(MissionState.ARMING)
        assert sm.arming_rejected("PREFLIGHT_FAIL: accel bias") is MissionState.PREFLIGHT
        assert "accel bias" in sm.history[-1].reason


class TestAbortRouting:
    """Aborts from every state, and the RTL-vs-land-now decision."""

    def test_abort_from_mission_goes_to_rtl(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.abort(AbortReason.OPERATOR) is MissionState.RTL
        assert sm.abort_reason is AbortReason.OPERATOR

    def test_abort_from_takeoff_goes_to_rtl(self):
        sm = fly_to(MissionState.TAKEOFF)
        assert sm.abort(AbortReason.GEOFENCE_BREACH) is MissionState.RTL

    def test_abort_from_rtl_stays_recovering(self):
        sm = fly_to(MissionState.RTL)
        assert sm.abort(AbortReason.LOW_BATTERY) is MissionState.RTL

    def test_critical_battery_lands_immediately(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.abort(AbortReason.CRITICAL_BATTERY) is MissionState.LANDING

    def test_ekf_divergence_lands_immediately(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.abort(AbortReason.EKF_DIVERGED) is MissionState.LANDING

    def test_rc_loss_uses_rtl_not_immediate_landing(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.abort(AbortReason.RC_LOSS) is MissionState.RTL

    def test_abort_during_landing_does_not_interrupt_the_descent(self):
        sm = fly_to(MissionState.LANDING)
        assert sm.abort(AbortReason.LOW_BATTERY) is MissionState.LANDING
        assert sm.abort_reason is AbortReason.LOW_BATTERY

    def test_abort_on_the_ground_from_idle_faults(self):
        sm = MissionStateMachine()
        assert sm.abort(AbortReason.OPERATOR) is MissionState.FAULT

    def test_abort_on_the_ground_from_preflight_faults(self):
        sm = fly_to(MissionState.PREFLIGHT)
        assert sm.abort(AbortReason.PREFLIGHT_FAILED) is MissionState.FAULT

    def test_abort_from_arming_faults(self):
        sm = fly_to(MissionState.ARMING)
        assert sm.abort(AbortReason.OPERATOR) is MissionState.FAULT

    def test_abort_from_disarmed_is_illegal(self):
        sm = fly_to(MissionState.DISARMED)
        with pytest.raises(IllegalTransition):
            sm.abort(AbortReason.OPERATOR)

    def test_abort_records_the_reason_in_history(self):
        sm = fly_to(MissionState.MISSION)
        sm.abort(AbortReason.GEOFENCE_BREACH, "45 m outside", now=12.0)
        reasons = [r.reason for r in sm.history]
        assert any("geofence_breach" in r and "45 m outside" in r for r in reasons)

    def test_abort_passes_through_the_abort_state(self):
        sm = fly_to(MissionState.MISSION)
        sm.abort(AbortReason.OPERATOR)
        assert ("MISSION", "ABORT") in sm.history_states()
        assert ("ABORT", "RTL") in sm.history_states()

    @pytest.mark.parametrize(
        "state", [MissionState.TAKEOFF, MissionState.MISSION, MissionState.RTL]
    )
    def test_abort_is_available_from_every_airborne_state(self, state):
        sm = fly_to(state)
        result = sm.abort(AbortReason.OPERATOR)
        assert result in (MissionState.RTL, MissionState.LANDING)

    def test_fault_helper(self):
        sm = fly_to(MissionState.MISSION)
        assert sm.fault("motor 3 desync") is MissionState.FAULT
        assert "motor 3 desync" in sm.history[-1].reason

    def test_reset_clears_the_abort_latch(self):
        sm = fly_to(MissionState.MISSION)
        sm.abort(AbortReason.CRITICAL_BATTERY)
        sm.confirm_disarmed()
        sm.reset()
        assert sm.abort_reason is None


class TestWaypointProgress:
    """Waypoint counters."""

    def test_begin_mission_resets_the_counters(self):
        sm = MissionStateMachine()
        sm.begin_mission(7)
        assert (sm.waypoint_index, sm.waypoint_count) == (0, 7)

    def test_advance_returns_true_while_waypoints_remain(self):
        sm = MissionStateMachine()
        sm.begin_mission(3)
        assert sm.advance_waypoint() is True
        assert sm.advance_waypoint() is True

    def test_advance_returns_false_on_the_last_waypoint(self):
        sm = MissionStateMachine()
        sm.begin_mission(3)
        sm.advance_waypoint()
        sm.advance_waypoint()
        assert sm.advance_waypoint() is False

    def test_index_never_runs_past_the_count(self):
        sm = MissionStateMachine()
        sm.begin_mission(2)
        for _ in range(10):
            sm.advance_waypoint()
        assert sm.waypoint_index == 2

    def test_mission_complete_flag(self):
        sm = MissionStateMachine()
        sm.begin_mission(2)
        assert not sm.mission_complete
        sm.advance_waypoint()
        sm.advance_waypoint()
        assert sm.mission_complete

    def test_empty_mission_is_never_complete(self):
        sm = MissionStateMachine()
        sm.begin_mission(0)
        assert not sm.mission_complete

    def test_negative_count_is_rejected(self):
        sm = MissionStateMachine()
        with pytest.raises(ValueError, match="waypoint_count must be >= 0"):
            sm.begin_mission(-1)
