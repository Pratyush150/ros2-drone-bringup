"""Mission executor state machine: explicit states, guarded transitions, abort paths.

The nominal flow is

    IDLE -> PREFLIGHT -> ARMING -> TAKEOFF -> MISSION -> RTL -> LANDING -> DISARMED

and every airborne state can also go to ABORT, which routes to RTL or LANDING
depending on how bad things are. ``FAULT`` is the terminal state for something
that cannot be recovered in the air.

Why a real state machine
------------------------
The alternative -- a pile of booleans and ``if armed and not landing and ...``
in a timer callback -- fails in exactly one way: some sequence of MAVROS
messages puts the flags in a combination nobody thought about, and the vehicle
takes off with a stale setpoint or refuses to land. Making states explicit and
transitions guarded means an illegal request is *rejected and logged*, not
silently half-applied.

Two rules this implementation enforces:

1. **No transition without a guard.** Every edge is declared in
   :data:`TRANSITIONS`. A request for an undeclared edge raises
   :class:`IllegalTransition` and leaves the state untouched.
2. **Preflight gates arming, not takeoff.** GPS fix, EKF convergence, battery,
   geofence load, and RC link are checked *before* the arm request goes out.
   Arming first and checking later is how you end up with a spinning vehicle
   you now have to disarm.

Pure logic. No ROS, no clocks you cannot inject -- pass ``now`` explicitly so
tests are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

__all__ = [
    "MissionState",
    "AbortReason",
    "IllegalTransition",
    "PreflightStatus",
    "VehicleSnapshot",
    "PreflightLimits",
    "check_preflight",
    "TRANSITIONS",
    "MissionStateMachine",
    "TransitionRecord",
    "AIRBORNE_STATES",
    "reachable_states",
]


class MissionState(str, Enum):
    """Every state the executor can be in."""

    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    MISSION = "MISSION"
    RTL = "RTL"
    LANDING = "LANDING"
    DISARMED = "DISARMED"
    ABORT = "ABORT"
    FAULT = "FAULT"


class AbortReason(str, Enum):
    """Why an abort was raised. Determines whether we RTL or land immediately."""

    OPERATOR = "operator"
    GEOFENCE_BREACH = "geofence_breach"
    LOW_BATTERY = "low_battery"
    CRITICAL_BATTERY = "critical_battery"
    RC_LOSS = "rc_loss"
    TELEMETRY_STALE = "telemetry_stale"
    EKF_DIVERGED = "ekf_diverged"
    PREFLIGHT_FAILED = "preflight_failed"


#: Abort reasons severe enough that climbing to RTL altitude is the wrong move.
#: A critical battery or a diverged estimator means put it down now, near here.
_LAND_NOW_REASONS: FrozenSet[AbortReason] = frozenset(
    {
        AbortReason.CRITICAL_BATTERY,
        AbortReason.EKF_DIVERGED,
    }
)


class IllegalTransition(RuntimeError):
    """Raised when a transition is not declared in :data:`TRANSITIONS`."""

    def __init__(self, current: MissionState, requested: MissionState) -> None:
        super().__init__(
            f"illegal transition {current.value} -> {requested.value}; "
            f"allowed from {current.value}: "
            f"{sorted(s.value for s in TRANSITIONS.get(current, frozenset()))}"
        )
        self.current = current
        self.requested = requested


#: The complete transition table. Anything not listed here is rejected.
TRANSITIONS: Dict[MissionState, FrozenSet[MissionState]] = {
    MissionState.IDLE: frozenset({MissionState.PREFLIGHT, MissionState.FAULT}),
    MissionState.PREFLIGHT: frozenset(
        {MissionState.ARMING, MissionState.IDLE, MissionState.ABORT, MissionState.FAULT}
    ),
    # Arming can fail back to PREFLIGHT (autopilot refused) or abort.
    MissionState.ARMING: frozenset(
        {
            MissionState.TAKEOFF,
            MissionState.PREFLIGHT,
            MissionState.ABORT,
            MissionState.FAULT,
        }
    ),
    MissionState.TAKEOFF: frozenset(
        {MissionState.MISSION, MissionState.ABORT, MissionState.FAULT}
    ),
    MissionState.MISSION: frozenset(
        {
            MissionState.RTL,
            MissionState.LANDING,
            MissionState.ABORT,
            MissionState.FAULT,
        }
    ),
    MissionState.RTL: frozenset(
        {MissionState.LANDING, MissionState.ABORT, MissionState.FAULT}
    ),
    MissionState.LANDING: frozenset(
        {MissionState.DISARMED, MissionState.ABORT, MissionState.FAULT}
    ),
    # ABORT is a decision point, not a resting state: it immediately routes.
    MissionState.ABORT: frozenset(
        {MissionState.RTL, MissionState.LANDING, MissionState.FAULT}
    ),
    # Terminal states.
    MissionState.DISARMED: frozenset({MissionState.IDLE}),
    MissionState.FAULT: frozenset({MissionState.IDLE}),
}

#: States in which the vehicle is (or may be) airborne.
AIRBORNE_STATES: FrozenSet[MissionState] = frozenset(
    {
        MissionState.TAKEOFF,
        MissionState.MISSION,
        MissionState.RTL,
        MissionState.LANDING,
        MissionState.ABORT,
    }
)


# --- preflight ---------------------------------------------------------------
@dataclass(frozen=True)
class PreflightLimits:
    """Thresholds the preflight gate checks against.

    Defaults are deliberately conservative for a small multirotor. Tune them for
    your airframe -- a 3D fix with 12 satellites is fine over open ground and
    not fine between buildings.
    """

    min_gps_fix_type: int = 3
    """3 = 3D fix in the MAVLink ``GPS_FIX_TYPE`` enum. 2 (2D) has no usable altitude."""

    min_satellites: int = 8
    min_battery_voltage: float = 14.0
    min_battery_percent: float = 0.40
    max_hdop: float = 2.0
    max_telemetry_age_s: float = 1.0
    require_rc: bool = True
    require_geofence: bool = True
    require_home_set: bool = True


@dataclass(frozen=True)
class VehicleSnapshot:
    """Everything preflight needs to know, sampled at one instant.

    Built by the ROS node from MAVROS topics; a plain value object here so the
    logic is testable without a middleware.
    """

    gps_fix_type: int = 0
    satellites: int = 0
    hdop: float = 99.0
    ekf_ok: bool = False
    battery_voltage: float = 0.0
    battery_percent: float = 0.0
    rc_connected: bool = False
    geofence_loaded: bool = False
    home_set: bool = False
    armed: bool = False
    telemetry_age_s: float = 99.0
    altitude_m: float = 0.0


@dataclass(frozen=True)
class PreflightStatus:
    """Result of a preflight evaluation."""

    passed: bool
    failures: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def summary(self) -> str:
        """One-line human summary, suitable for a diagnostic message."""
        if self.passed:
            return (
                "preflight OK"
                if not self.warnings
                else f"preflight OK with warnings: {'; '.join(self.warnings)}"
            )
        return f"preflight FAILED: {'; '.join(self.failures)}"


def check_preflight(
    snapshot: VehicleSnapshot, limits: Optional[PreflightLimits] = None
) -> PreflightStatus:
    """Evaluate every preflight gate and report *all* failures, not just the first.

    Reporting all of them matters on a field bench: you want one trip to the
    vehicle, not five.

    Args:
        snapshot: Current vehicle state.
        limits: Thresholds; defaults to :class:`PreflightLimits`.

    Returns:
        A :class:`PreflightStatus` with an explicit list of failure strings.
    """
    lim = limits or PreflightLimits()
    failures: List[str] = []
    warnings: List[str] = []

    if snapshot.telemetry_age_s > lim.max_telemetry_age_s:
        failures.append(
            f"telemetry stale: {snapshot.telemetry_age_s:.2f}s > "
            f"{lim.max_telemetry_age_s:.2f}s (everything below is unreliable)"
        )
    if snapshot.gps_fix_type < lim.min_gps_fix_type:
        failures.append(
            f"GPS fix type {snapshot.gps_fix_type} < required {lim.min_gps_fix_type}"
        )
    if snapshot.satellites < lim.min_satellites:
        failures.append(
            f"only {snapshot.satellites} satellites, need {lim.min_satellites}"
        )
    if snapshot.hdop > lim.max_hdop:
        failures.append(f"HDOP {snapshot.hdop:.2f} > {lim.max_hdop:.2f}")
    if not snapshot.ekf_ok:
        failures.append("EKF not converged / attitude+position estimate not valid")
    if snapshot.battery_voltage < lim.min_battery_voltage:
        failures.append(
            f"battery {snapshot.battery_voltage:.2f} V < "
            f"{lim.min_battery_voltage:.2f} V"
        )
    if snapshot.battery_percent < lim.min_battery_percent:
        failures.append(
            f"battery {snapshot.battery_percent * 100:.0f}% < "
            f"{lim.min_battery_percent * 100:.0f}%"
        )
    if lim.require_rc and not snapshot.rc_connected:
        failures.append("no RC link (manual takeover would be impossible)")
    if lim.require_geofence and not snapshot.geofence_loaded:
        failures.append("geofence not loaded")
    if lim.require_home_set and not snapshot.home_set:
        failures.append("home position not set (RTL would have nowhere to go)")
    if snapshot.armed:
        warnings.append("vehicle is already armed before preflight completed")

    return PreflightStatus(
        passed=not failures, failures=tuple(failures), warnings=tuple(warnings)
    )


# --- transition log ----------------------------------------------------------
@dataclass(frozen=True)
class TransitionRecord:
    """One accepted transition, for the flight log."""

    from_state: MissionState
    to_state: MissionState
    reason: str
    timestamp: float


# --- the machine -------------------------------------------------------------
@dataclass
class MissionStateMachine:
    """Guarded executor state machine.

    Example:
        >>> sm = MissionStateMachine()
        >>> sm.state
        <MissionState.IDLE: 'IDLE'>
        >>> sm.request(MissionState.PREFLIGHT, "operator start")
        <MissionState.PREFLIGHT: 'PREFLIGHT'>
        >>> sm.request(MissionState.MISSION, "skip ahead")
        Traceback (most recent call last):
        ...
        drone_bringup.core.state_machine.IllegalTransition: illegal transition ...
    """

    state: MissionState = MissionState.IDLE
    limits: PreflightLimits = field(default_factory=PreflightLimits)
    history: List[TransitionRecord] = field(default_factory=list)
    abort_reason: Optional[AbortReason] = None
    last_preflight: Optional[PreflightStatus] = None
    waypoint_index: int = 0
    waypoint_count: int = 0
    on_transition: Optional[Callable[[TransitionRecord], None]] = None

    # -- queries ------------------------------------------------------------
    def can(self, target: MissionState) -> bool:
        """True if ``target`` is a declared transition from the current state."""
        return target in TRANSITIONS.get(self.state, frozenset())

    @property
    def allowed(self) -> FrozenSet[MissionState]:
        """The set of states reachable in one step from here."""
        return TRANSITIONS.get(self.state, frozenset())

    @property
    def is_airborne(self) -> bool:
        """True if the current state implies the vehicle may be in the air."""
        return self.state in AIRBORNE_STATES

    @property
    def is_terminal(self) -> bool:
        """True for states that end a sortie."""
        return self.state in (MissionState.DISARMED, MissionState.FAULT)

    # -- core transition ----------------------------------------------------
    def request(
        self, target: MissionState, reason: str = "", now: float = 0.0
    ) -> MissionState:
        """Attempt a transition.

        Args:
            target: Desired state.
            reason: Free text recorded in the history; show it in the log.
            now: Timestamp to record. Injected so tests are deterministic.

        Returns:
            The new state.

        Raises:
            IllegalTransition: If the edge is not declared. The state is left
                unchanged, so a caller that catches this can keep flying.
        """
        if target not in TRANSITIONS.get(self.state, frozenset()):
            raise IllegalTransition(self.state, target)
        record = TransitionRecord(self.state, target, reason, now)
        self.state = target
        self.history.append(record)
        if target is not MissionState.ABORT:
            # Clear the latch once we have routed out of ABORT.
            if self.state in (MissionState.IDLE, MissionState.DISARMED):
                self.abort_reason = None
        if self.on_transition is not None:
            self.on_transition(record)
        return self.state

    def try_request(
        self, target: MissionState, reason: str = "", now: float = 0.0
    ) -> bool:
        """Non-raising :meth:`request`. Returns True if the transition happened."""
        try:
            self.request(target, reason, now)
        except IllegalTransition:
            return False
        return True

    # -- high-level operations ---------------------------------------------
    def start_preflight(self, now: float = 0.0) -> MissionState:
        """IDLE -> PREFLIGHT."""
        return self.request(MissionState.PREFLIGHT, "operator start", now)

    def run_preflight(
        self, snapshot: VehicleSnapshot, now: float = 0.0
    ) -> PreflightStatus:
        """Evaluate preflight and advance to ARMING on pass, ABORT on fail.

        Must be called from :attr:`MissionState.PREFLIGHT`.

        Raises:
            IllegalTransition: If called from any other state.
        """
        if self.state is not MissionState.PREFLIGHT:
            raise IllegalTransition(self.state, MissionState.PREFLIGHT)
        status = check_preflight(snapshot, self.limits)
        self.last_preflight = status
        if status.passed:
            self.request(MissionState.ARMING, "preflight passed", now)
        else:
            self.abort(AbortReason.PREFLIGHT_FAILED, status.summary(), now)
        return status

    def confirm_armed(self, now: float = 0.0) -> MissionState:
        """ARMING -> TAKEOFF, once the autopilot reports armed."""
        return self.request(MissionState.TAKEOFF, "autopilot reported armed", now)

    def arming_rejected(self, detail: str = "", now: float = 0.0) -> MissionState:
        """ARMING -> PREFLIGHT after the autopilot refused the arm command.

        This is a normal outcome, not a fault: PX4 refuses arming for a long
        list of its own preflight reasons. Go back and re-check rather than
        retrying the same command in a loop.
        """
        text = f"arming rejected: {detail}" if detail else "arming rejected"
        return self.request(MissionState.PREFLIGHT, text, now)

    def confirm_takeoff_complete(self, now: float = 0.0) -> MissionState:
        """TAKEOFF -> MISSION once the target altitude is reached."""
        return self.request(MissionState.MISSION, "takeoff altitude reached", now)

    def begin_mission(self, waypoint_count: int) -> None:
        """Reset waypoint progress counters for a newly loaded mission."""
        if waypoint_count < 0:
            raise ValueError(f"waypoint_count must be >= 0, got {waypoint_count}")
        self.waypoint_index = 0
        self.waypoint_count = waypoint_count

    def advance_waypoint(self) -> bool:
        """Mark the current waypoint reached.

        Returns:
            True if there are more waypoints to fly, False if the mission is done.
        """
        self.waypoint_index = min(self.waypoint_index + 1, self.waypoint_count)
        return self.waypoint_index < self.waypoint_count

    @property
    def mission_complete(self) -> bool:
        """True once every waypoint has been reached."""
        return self.waypoint_count > 0 and self.waypoint_index >= self.waypoint_count

    def finish_mission(self, now: float = 0.0) -> MissionState:
        """MISSION -> RTL at the end of the waypoint list."""
        return self.request(MissionState.RTL, "mission complete", now)

    def begin_landing(self, now: float = 0.0) -> MissionState:
        """-> LANDING, from MISSION, RTL or ABORT."""
        return self.request(MissionState.LANDING, "descending to land", now)

    def confirm_disarmed(self, now: float = 0.0) -> MissionState:
        """LANDING -> DISARMED once the autopilot reports disarmed."""
        return self.request(MissionState.DISARMED, "touchdown, disarmed", now)

    def reset(self, now: float = 0.0) -> MissionState:
        """DISARMED or FAULT -> IDLE, ready for the next sortie."""
        state = self.request(MissionState.IDLE, "reset", now)
        self.abort_reason = None
        self.waypoint_index = 0
        self.waypoint_count = 0
        return state

    # -- abort --------------------------------------------------------------
    def abort(
        self, reason: AbortReason, detail: str = "", now: float = 0.0
    ) -> MissionState:
        """Enter ABORT and immediately route to the right recovery state.

        Routing:

        * On the ground (IDLE / PREFLIGHT / ARMING): ABORT then straight to
          FAULT -- there is nothing to recover, and the operator must clear it.
        * Airborne with a *critical* reason (critical battery, EKF diverged):
          land immediately where we are. Climbing to RTL altitude on a dying
          battery is how you turn a forced landing into a crash.
        * Airborne otherwise: RTL.
        * Already LANDING: stay landing; there is no better option than
          finishing the descent.

        Raises:
            IllegalTransition: If ABORT is not reachable (already DISARMED or
                FAULT).
        """
        self.abort_reason = reason
        text = f"abort({reason.value})" + (f": {detail}" if detail else "")

        if self.state is MissionState.LANDING:
            # Already committed to the ground; do not interrupt the descent.
            return self.state

        if self.state in (MissionState.IDLE, MissionState.PREFLIGHT, MissionState.ARMING):
            if self.state is MissionState.IDLE:
                return self.request(MissionState.FAULT, text, now)
            self.request(MissionState.ABORT, text, now)
            return self.request(MissionState.FAULT, f"{text} (on ground)", now)

        self.request(MissionState.ABORT, text, now)
        if reason in _LAND_NOW_REASONS:
            return self.request(MissionState.LANDING, f"{text} -> land now", now)
        return self.request(MissionState.RTL, f"{text} -> return to launch", now)

    def fault(self, detail: str, now: float = 0.0) -> MissionState:
        """Force the terminal FAULT state from anywhere it is reachable."""
        return self.request(MissionState.FAULT, f"fault: {detail}", now)

    # -- introspection ------------------------------------------------------
    def history_states(self) -> List[Tuple[str, str]]:
        """History as ``(from, to)`` value pairs -- handy in tests and logs."""
        return [(r.from_state.value, r.to_state.value) for r in self.history]

    def snapshot(self) -> Dict[str, object]:
        """Serialisable view of the machine, for a status topic."""
        return {
            "state": self.state.value,
            "abort_reason": None if self.abort_reason is None else self.abort_reason.value,
            "waypoint_index": self.waypoint_index,
            "waypoint_count": self.waypoint_count,
            "airborne": self.is_airborne,
            "preflight_passed": None
            if self.last_preflight is None
            else self.last_preflight.passed,
        }


def reachable_states(start: MissionState = MissionState.IDLE) -> FrozenSet[MissionState]:
    """Breadth-first closure of :data:`TRANSITIONS` from ``start``.

    Used by the tests to assert that no state is stranded and that every state
    has a path back to a terminal state.
    """
    seen = {start}
    queue: List[MissionState] = [start]
    while queue:
        current = queue.pop()
        for nxt in TRANSITIONS.get(current, frozenset()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return frozenset(seen)
