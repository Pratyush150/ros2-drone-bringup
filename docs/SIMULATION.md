# Simulation: PX4 SITL, Gazebo, and this package

Everything here is about getting a PX4 SITL vehicle talking to `drone_bringup`
over MAVROS, and about the port layout that decides whether it works.

---

## 1. The port layout, and why it matters more than anything else

PX4 SITL opens **two independent MAVLink streams**. They carry the same
protocol but exist for different consumers:

| Port | Name in PX4 config | Who connects | What breaks if you get it wrong |
|---|---|---|---|
| `udp://:14540` | offboard / onboard API | MAVSDK, MAVROS, your autonomy code | Your offboard setpoints go nowhere |
| `udp://:14550` | GCS | QGroundControl, Mission Planner | QGC never finds the vehicle |

**Do not point two things at the same UDP port.**

A UDP socket bound without `SO_REUSEADDR` refuses the second binder outright.
With `SO_REUSEADDR` (which several MAVLink tools set) both processes bind
successfully and then the kernel hands each datagram to exactly one of them —
usually the most recent binder, non-deterministically in some kernels. The
symptom is not an error. It is one of:

- QGroundControl sits on "Waiting for vehicle connection" while MAVROS is happy.
- MAVROS reports `connected: false` while QGC flies the vehicle fine.
- Both appear to work, and telemetry rates are visibly halved because each
  process is receiving roughly every other packet.

You will spend an hour on this. The fix is trivial: **MAVROS gets 14540, QGC
gets 14550, and if you need a third consumer you run
[mavlink-router](https://github.com/mavlink-router/mavlink-router) or
`mavproxy` and fan out from there.** Never double-bind.

`launch/sitl.launch.py` sets `fcu_url` to `udp://:14540@127.0.0.1:14557` — bind
locally on the offboard port, reply to PX4's simulator endpoint — and leaves
14550 alone. There is a test in `tests/test_package_integrity.py` that fails if
anyone changes that to a URL containing 14550.

**Multi-vehicle**: PX4 offsets both by the instance index. Vehicle `N` uses
`14540 + N` and `14550 + N`. The `instance` launch argument applies that offset.

---

## 2. Prerequisites

- ROS 2 Humble, sourced.
- MAVROS: `sudo apt install ros-humble-mavros ros-humble-mavros-extras`
- The MAVROS geoid dataset — **install this or altitude will be wrong by tens
  of metres** and every AMSL number you look at is nonsense:

  ```bash
  wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
  sudo bash ./install_geographiclib_datasets.sh
  ```

- A built PX4-Autopilot checkout with Gazebo (gz-sim / Harmonic) support:

  ```bash
  git clone https://github.com/PX4/PX4-Autopilot.git --recursive
  cd PX4-Autopilot && bash ./Tools/setup/ubuntu.sh
  make px4_sitl gz_x500          # first build takes a while
  ```

---

## 3. Quickstart

Build and source this package:

```bash
cd ~/ros2_ws && colcon build --packages-select drone_bringup
source install/setup.bash
```

Make the shipped world and model visible to Gazebo:

```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:\
$(ros2 pkg prefix drone_bringup)/share/drone_bringup/worlds:\
$(ros2 pkg prefix drone_bringup)/share/drone_bringup/models
```

### Option A — one command

```bash
ros2 launch drone_bringup sitl.launch.py px4_dir:=$HOME/PX4-Autopilot rviz:=true
```

### Option B — two terminals (recommended while iterating)

Terminal 1, PX4 + Gazebo:

```bash
cd ~/PX4-Autopilot
PX4_GZ_WORLD=survey_field make px4_sitl gz_x500
```

Terminal 2, MAVROS + bringup:

```bash
ros2 launch drone_bringup sitl.launch.py start_px4:=false
```

Restarting the ROS side without rebuilding the simulator saves a lot of time,
and PX4's own console stays readable.

### Verify the link before doing anything else

```bash
ros2 topic echo /mavros/state --once
# connected: true, mode: AUTO.LOITER, armed: false
```

If `connected` is `false`, it is the port. Check nothing else is bound:

```bash
ss -ulpn | grep -E '1454[0-9]|1455[0-9]'
```

### Run the mission

```bash
ros2 topic echo /preflight_check/report          # watch the gate
ros2 topic pub --once /mission_executor/command std_msgs/String "{data: start}"
ros2 topic echo /mission_executor/state
```

In SITL there is no RC transmitter, so set `require_rc: false` in
`config/preflight_check.yaml` or preflight will (correctly) refuse to arm.

---

## 4. Things that go wrong in SITL and what they actually mean

| Symptom | Cause |
|---|---|
| `ros2 topic echo` prints nothing on a topic that exists | QoS mismatch. `echo` defaults to RELIABLE; sensor topics are BEST_EFFORT. Use `ros2 topic echo --qos-reliability best_effort`. See `drone_bringup/nodes/qos.py`. |
| MAVROS connects, then drops every few seconds | Two things bound to the same port. |
| Vehicle refuses OFFBOARD | Setpoints were not already streaming. PX4 needs a couple of seconds of stream before it will accept the mode. The executor handles this; if you are testing by hand, publish setpoints first, then request the mode. |
| Vehicle enters OFFBOARD then immediately exits | Setpoint stream stalled for >0.5 s. Usually a blocking call inside a timer callback starving the executor. |
| Timestamps all zero, staleness checks never fire | `use_sim_time:=true` with nothing publishing `/clock`, or `false` while Gazebo is. |
| Position drifts steadily in a fresh sim | The EKF has not converged yet. Wait for `local_position/pose` to settle before arming. |
| Real-time factor collapses to 0.2 | Sensor rendering. Drop camera resolution/FPS in the world, or run headless with `HEADLESS=1`. |

---

## 5. Gazebo vs AirSim vs Isaac Sim

Honest comparison for drone work. There is no winner; there is a right tool per
question you are trying to answer.

| | **Gazebo (gz-sim / Harmonic)** | **AirSim / Colosseum** | **NVIDIA Isaac Sim** |
|---|---|---|---|
| **Physics fidelity** | Good rigid-body (DART/Bullet). Aerodynamics come from plugins and are simple — fine for control and mission logic, not for airframe design. | Similar rigid-body quality; the multirotor model is reasonable. Physics is not the reason people pick it. | PhysX 5, best-in-class contact and articulation. Overkill for a free-flying multirotor; genuinely useful for manipulation and landing-gear contact. |
| **Sensor / rendering realism** | Functional. Ogre2 rendering is adequate for geometry-driven perception, weak for anything that depends on materials, lighting, or lens behaviour. | Strong. Unreal Engine rendering, weather and time-of-day, decent camera models. Built for vision work. | Strongest. RTX path tracing, physically-based materials, real sensor models (rolling shutter, motion blur, lidar beam divergence, radar). |
| **GPU cost** | Runs headless on CPU. A laptop iGPU handles a multirotor and a couple of cameras. | Needs a real GPU. Expect 4–6 GB VRAM for a modest scene. | Heavy. RTX-class GPU, 8 GB VRAM minimum and you will want more. Not usable in a typical CI runner. |
| **PX4 / ArduPilot integration** | First-class. PX4 ships and maintains the gz-sim bridge; ArduPilot SITL support is mature. This is the path of least resistance. | AirSim has a PX4 HITL/SITL bridge but it has lagged; the original repo is archived and Colosseum is the community fork. Expect integration work. | PX4 integration exists via Pegasus and the Isaac ROS stack but is younger and moves fast. Budget time. |
| **ROS 2 integration** | Native `ros_gz_bridge`, Humble-supported, well-trodden. | ROS 2 wrapper exists, quality varies by fork. | Native ROS 2 bridge, actively developed, tied to specific Isaac releases. |
| **Maturity / ecosystem** | Most examples, most answers, most existing worlds. Boring in the best sense. | Was the go-to for drone vision 2018–2022. Microsoft archived AirSim in 2022; Colosseum is community-maintained and less active than it was. | Newest, fastest-moving, best-funded. Also the most likely to break between releases. |
| **Synthetic data / domain randomisation** | Manual. You script it yourself. | Good built-in support: weather, segmentation, depth, object poses. | Excellent — Replicator is purpose-built for this, with ground-truth annotation out of the box. |
| **Headless / CI** | Yes, cheaply. This is the only one you should plan to run in CI. | Possible but expensive. | Possible; expensive and GPU-gated. |

**How to choose:**

- **Flight logic, mission execution, failsafes, ROS plumbing** → Gazebo. It is
  what PX4 tests against, it runs headless, and none of those questions need
  photorealism. That is what this package targets.
- **Vision-based perception that must survive real lighting** → AirSim/Colosseum
  or Isaac Sim. Gazebo's renderer will make your detector look better than it is.
- **Sim-to-real transfer, synthetic training data, sensor-level realism, or
  contact-rich tasks** → Isaac Sim, and budget for the GPU.

Nothing in `drone_bringup/core` depends on the simulator. If you move to Isaac
Sim or AirSim, the mission format, geofence, geodesy, and state machine come
with you; only the launch files change.

---

## 6. Adding this package's world

`worlds/survey_field.sdf` is a 400 × 400 m field with a landing pad, a 45 m
comms mast, and a few obstacles. Its `<spherical_coordinates>` matches the
origin in `config/example_mission.yaml` and `config/example_geofence.yaml`.

**They must match.** PX4's simulated GPS derives its lat/lon from the world's
spherical coordinates. Change one without the other and every geodetic waypoint
in the mission lands somewhere else entirely — the vehicle flies a perfectly
correct pattern, in the wrong hemisphere.

```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/path/to/drone_bringup/worlds:/path/to/drone_bringup/models
PX4_GZ_WORLD=survey_field make px4_sitl gz_x500
```

The world sets a light 2.5 m/s wind. Leave it on. Zero wind is the least useful
simulator setting there is: it hides every controller tuning problem you will
meet the first time you fly outdoors.
