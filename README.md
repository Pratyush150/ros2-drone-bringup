# ros2-drone-bringup

ROS 2 Humble bringup for a PX4 or ArduPilot multirotor: MAVROS telemetry
normalisation, a guarded mission-executor state machine, a predictive geofence,
preflight gating, and PX4 SITL + Gazebo launch files.

## The problem

Standing up a ROS 2 stack on a drone is not hard because the algorithms are
hard. It is hard because of a specific set of things that fail quietly:

- **A topic that publishes nothing.** ROS 2 requires *QoS compatibility*, not
  just a matching name and type. A RELIABLE subscriber never connects to a
  BEST_EFFORT publisher, with no error anywhere. `ros2 topic list` still shows
  the topic. This is the single most common ROS 2 bringup failure.
- **Frames.** ROS is ENU/FLU. PX4 is NED/FRD. The conversion is a two-part
  composition, and applying only half of it gives you a vehicle that hovers
  correctly and flies 90 degrees off heading.
- **Altitude datums.** GPS reports ellipsoidal height, your mission says "40 m",
  your geofence says "120 m", and three of those are measured from different
  places.
- **Two things bound to the same MAVLink port.** MAVROS and QGroundControl both
  pointed at 14550: one of them silently goes quiet, and which one is up to
  your kernel.
- **A pile of booleans pretending to be a state machine.** `if armed and not
  landing and gps_ok` works until some ordering of MAVROS messages produces a
  combination nobody considered.
- **Telemetry that stopped updating.** A stale topic and a topic with an
  unchanged value are byte-identical. Nothing tells you which one you have.

This package is the skeleton that handles those, so you can get to the part you
actually care about.

## What you get

| | |
|---|---|
| `drone_bringup/core/geodesy.py` | WGS84 <-> local ENU/NED, haversine, bearing, `LocalOrigin`. Explicit ENU/NED converters with the convention clash documented. |
| `drone_bringup/core/frames.py` | Quaternion/Euler, FRD<->FLU, and the exact PX4<->ROS attitude transform chain. |
| `drone_bringup/core/mission.py` | YAML mission format: takeoff, waypoints, orbit, RTL, land, and real polygon-fill lawnmower survey generation. Validation errors name the item and the field. |
| `drone_bringup/core/geofence.py` | Inclusion/exclusion polygons + altitude band, signed distance to boundary, and **time-to-breach** from the current velocity. |
| `drone_bringup/core/state_machine.py` | Guarded executor state machine with abort paths from every state and preflight gating. Illegal transitions are rejected, not applied. |
| `drone_bringup/nodes/` | Four rclpy nodes wiring the above to MAVROS. |
| `launch/` | SITL, hardware, and mission-only launch, with a namespace for multi-vehicle. |
| `worlds/`, `models/` | A Gazebo survey field and a landing pad. Text SDF, no binaries. |
| `tests/` | 8 files, 528 test cases, 446 assertions. Pure Python, no ROS needed. |

**Every line of flight logic lives in `drone_bringup/core/`, which imports no
ROS at all.** That is enforced by a test. It means the geometry, the mission
format, the geofence, and the state machine are unit-tested in CI on a machine
with no ROS installation, and reviewable in isolation.

## Quickstart (SITL)

```bash
# 1. Build
cd ~/ros2_ws/src && git clone <this repo> drone_bringup
cd ~/ros2_ws && colcon build --packages-select drone_bringup && source install/setup.bash

# 2. Make the shipped world visible to Gazebo
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:\
$(ros2 pkg prefix drone_bringup)/share/drone_bringup/worlds:\
$(ros2 pkg prefix drone_bringup)/share/drone_bringup/models

# 3. PX4 SITL + Gazebo + MAVROS + bringup
ros2 launch drone_bringup sitl.launch.py px4_dir:=$HOME/PX4-Autopilot rviz:=true

# 4. Confirm the link BEFORE anything else
ros2 topic echo /mavros/state --once     # connected: true

# 5. Watch the preflight gate, then start
ros2 topic echo /preflight_check/report
ros2 topic pub --once /mission_executor/command std_msgs/String "{data: start}"
```

There is no RC transmitter in SITL, so set `require_rc: false` in
`config/preflight_check.yaml` or preflight will correctly refuse to arm.

Run the core tests with no ROS at all:

```bash
python3 -m pytest -q          # 528 passed
```

Full simulation notes, the port layout, and a Gazebo/AirSim/Isaac Sim
comparison: [`docs/SIMULATION.md`](docs/SIMULATION.md).
TF tree and the static-identity trap: [`docs/TF_TREE.md`](docs/TF_TREE.md).

## Architecture

```
   PX4 / ArduPilot  (NED world, FRD body, scalar-first quaternions)
          |
          |  MAVLink   udp://:14540 offboard   |   udp://:14550 QGC
          v                                     (never both to one port)
       MAVROS   -- converts to ENU/FLU, publishes /mavros/**
          |
  ========|=========================================================
          |                    drone_bringup
          |
    +-----+---------------+---------------------+
    |                     |                     |
    v                     v                     v
 telemetry_bridge   preflight_check      geofence_monitor
 - staleness             - GPS/EKF/batt      - loads geofence.yaml
 - normalised odom       - RC / home         - point-in-polygon
 - diagnostics           - geofence loaded   - TIME TO BREACH
    |                     |                     |
    |                     | ~/ready (latched)   | ~/breach (latched)
    |                     v                     v
    |              +--------------------------------+
    +------------> |       mission_executor         |
                   |                                |
                   |  MissionStateMachine           |
                   |  IDLE -> PREFLIGHT -> ARMING   |
                   |   -> TAKEOFF -> MISSION        |
                   |   -> RTL -> LANDING -> DISARMED|
                   |         \                      |
                   |          -> ABORT -> RTL|LAND  |
                   +--------------------------------+
                        |                |
                        | setpoints      | service calls
                        v                v
                  /mavros/setpoint_position/local
                  /mavros/cmd/arming, /mavros/set_mode, /mavros/cmd/takeoff

  ================ drone_bringup/core (NO ROS IMPORTS) ==============
     geodesy.py   frames.py   mission.py   geofence.py   state_machine.py
     WGS84<->ENU  PX4<->ROS   YAML+survey  polygons +    guarded
     /NED         attitude    grid gen     time-to-      transitions
                                           breach        + preflight
```

## Nodes and topics

### `telemetry_bridge`

MAVROS is the only thing this node knows about. Everything downstream talks to
its outputs, so a MAVROS upgrade does not rewrite your graph.

| Direction | Topic | Type | QoS |
|---|---|---|---|
| sub | `<mavros_ns>/state` | `mavros_msgs/State` | reliable, transient local |
| sub | `<mavros_ns>/extended_state` | `mavros_msgs/ExtendedState` | reliable, transient local |
| sub | `<mavros_ns>/global_position/global` | `sensor_msgs/NavSatFix` | best effort |
| sub | `<mavros_ns>/local_position/pose` | `geometry_msgs/PoseStamped` | best effort |
| sub | `<mavros_ns>/local_position/velocity_local` | `geometry_msgs/TwistStamped` | best effort |
| sub | `<mavros_ns>/imu/data` | `sensor_msgs/Imu` | best effort |
| sub | `<mavros_ns>/battery` | `sensor_msgs/BatteryState` | best effort |
| pub | `~/odom` | `nav_msgs/Odometry` | best effort |
| pub | `~/status` | `std_msgs/String` | reliable, transient local |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | reliable |

### `preflight_check`

| Direction | Topic | Type |
|---|---|---|
| sub | `<mavros_ns>/state`, `.../gpsstatus/gps1/raw`, `.../battery`, `.../local_position/pose`, `.../rc/in`, `.../home_position/home` | MAVROS |
| sub | `/geofence_monitor/status` | `std_msgs/String` |
| pub | `~/ready` | `std_msgs/Bool` (latched) |
| pub | `~/report` | `std_msgs/String` (latched) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` |

Gates: 3D GPS fix, satellite count, HDOP, EKF converged, battery voltage **and**
percentage, RC link, geofence loaded, home set, telemetry freshness. It reports
*every* failure at once, because you want one trip to the vehicle, not five.

### `geofence_monitor`

| Direction | Topic | Type |
|---|---|---|
| sub | `<mavros_ns>/global_position/global`, `.../local_position/pose`, `.../local_position/velocity_local` | MAVROS |
| pub | `~/status` | `std_msgs/String` (latched: `loaded: N zones` or `error: ...`) |
| pub | `~/breach` | `std_msgs/Bool` (latched) |
| pub | `~/margin` | `std_msgs/Float32` |
| pub | `~/markers` | `visualization_msgs/MarkerArray` (latched, for RViz) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` |

It publishes a verdict and never commands the vehicle. The executor decides what
to do, which means you can run the monitor in advisory mode on a manually flown
vehicle without it touching the control path.

### `mission_executor`

| Direction | Topic | Type |
|---|---|---|
| sub | `<mavros_ns>/state`, `.../local_position/pose`, `.../global_position/global` | MAVROS |
| sub | `/preflight_check/ready`, `/geofence_monitor/breach` | `std_msgs/Bool` |
| sub | `~/command` | `std_msgs/String` — `start` \| `abort` \| `rtl` \| `reset` |
| pub | `<mavros_ns>/setpoint_position/local` | `geometry_msgs/PoseStamped` |
| pub | `~/state` | `std_msgs/String` (latched) |
| pub | `~/path` | `nav_msgs/Path` (latched, for RViz) |
| pub | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` |
| srv | `<mavros_ns>/cmd/arming`, `/set_mode`, `/cmd/takeoff`, `/cmd/land` | MAVROS |

## Mission format

```yaml
name: irchel_survey
origin: {latitude: 47.397742, longitude: 8.545594, altitude: 488.0}
default_speed: 5.0
rtl_altitude: 60.0
items:
  - {type: takeoff, altitude: 30.0}
  - {type: waypoint, latitude: 47.3981, longitude: 8.5463, altitude: 35.0,
     speed: 8.0, acceptance_radius: 3.0, hold_time: 2.0}
  - type: survey                 # boustrophedon polygon fill
    altitude: 40.0
    spacing: 20.0                # sensor swath, metres
    heading: 90.0                # deg clockwise from north; 90 = east-west lines
    line_extension: 10.0         # overshoot so the vehicle is settled on-line
    polygon: [[lat, lon], ...]
  - {type: orbit, latitude: ..., longitude: ..., altitude: 35.0,
     radius: 25.0, turns: 1.0, clockwise: true}
  - {type: rtl}
```

**All altitudes are metres above the mission origin.** Not AMSL, not
ellipsoidal. One datum, stated once, used everywhere.

Validation is eager and located: `mission.items[3] (survey).spacing: must be
> 0, got -5.0`. A mission that half-loads and then fails at item 14 mid-flight
is much worse than one that refuses to load on the bench.

The survey generator is real scanline polygon fill in the local ENU plane. It
handles concave polygons (a U-shaped field produces two segments per scan, not
one line through the notch), alternates direction so the vehicle never deadheads,
and is tested for coverage against sampled interior points.

## Geofence format

```yaml
origin: {latitude: 47.397742, longitude: 8.545594, altitude: 488.0}
max_altitude_m: 120.0
min_altitude_m: -10.0
zones:
  - {name: operating_area, kind: inclusion, vertices: [[lat, lon], ...]}
  - {name: mast_keepout,   kind: exclusion, max_altitude_m: 50.0,
     vertices: [[lat, lon], ...]}
```

The number worth acting on is **time to breach**, not "am I outside". A 15 m/s
multirotor decelerating at 3 m/s² needs 5 s and 37 m to stop, plus link
latency. A binary inside/outside check fires when it is already too late.

## What this is / isn't

**Is:**

- A working bringup skeleton for a PX4 or ArduPilot multirotor on ROS 2 Humble.
- Correct, tested implementations of the things people get wrong: frame
  conversions, geodetic projection, geofence prediction, guarded state
  transitions, QoS choices.
- A mission format with validation that tells you what is wrong and where.
- A place to hang your own perception, planning, or control.

**Isn't:**

- Tuned for your airframe. Every threshold in `config/` — battery voltage,
  satellite count, HDOP, acceptance radius, tick rate — is a default for a 4S
  sub-2 kg multirotor over open ground. Read them and change them.
- A flight controller or a control law. PX4/ArduPilot fly the vehicle; this
  package tells them where to go and when to stop.
- An obstacle avoidance system. Legs between waypoints are straight lines. If a
  leg has to avoid something, you add the waypoint that makes it avoid it — the
  shipped example mission does exactly that around its keepout zone.
- A replacement for your own risk assessment. The geofence is a software check
  in a companion computer, downstream of a radio link and an estimator. Set the
  autopilot's own geofence parameters too.
- Certified, qualified, or flight-proven for anything. It is engineering
  scaffolding. Test it in SITL, then test it on a tether, then fly it.

**Known limitations:**

- The tangent-plane projection is exact enough for any site you would fly a
  multirotor over; it is not a substitute for a proper geodetic library over
  tens of kilometres.
- Nodes are lifecycle-*shaped*, not `LifecycleNode` subclasses. Converting them
  is a mechanical refactor (`configure()` -> `on_configure`), deliberately left
  as a refactor rather than a rewrite.
- No multi-vehicle coordination. The namespace argument gives you N independent
  stacks, not N cooperating ones.
- ArduPilot support means "the MAVROS topic names and services are the same".
  Mode strings differ (`GUIDED` vs `OFFBOARD`); change them in
  `mission_executor_node.py`.

## Layout

```
drone_bringup/
  core/                 no ROS imports, ever (enforced by a test)
    geodesy.py  frames.py  mission.py  geofence.py  state_machine.py
  nodes/                rclpy nodes
    qos.py              shared QoS profiles + the QoS-mismatch explainer
    mission_executor_node.py  telemetry_bridge_node.py
    geofence_monitor_node.py  preflight_check_node.py
launch/    bringup, sitl, hardware, mission
config/    per-node params, example mission, example geofence, rviz
worlds/    survey_field.sdf
models/    landing_pad/
docs/      SIMULATION.md, TF_TREE.md
tests/     8 files, pure pytest, no ROS
```

## Related

- [`px4-mavlink-companion`](https://github.com/Pratyush150/px4-mavlink-companion) —
  MAVLink bridge, link watchdog, and offboard control between an FC and a
  companion computer, without ROS in the loop.
- [`drone-control-toolkit`](https://github.com/Pratyush150/drone-control-toolkit) —
  PID/LQR/complementary-filter/EKF control and estimation with a sim harness.
- [`flight-log-analyzer`](https://github.com/Pratyush150/flight-log-analyzer) —
  PX4 ULog / ArduPilot log analysis: vibration, EKF, power, mode timeline.
- [`lidar-slam-toolkit`](https://github.com/Pratyush150/lidar-slam-toolkit) —
  LIO-SAM / Cartographer configs plus drift and extrinsics diagnostics.

## License

MIT. See [LICENSE](LICENSE).
