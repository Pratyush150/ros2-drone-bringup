# The TF tree for a drone, and the shortcut that ruins it

## The tree

```
                    earth                      (optional, only for multi-robot
                      |                         or geo-referenced setups)
                      |  static, from the GPS origin
                      v
                     map                       fixed, gravity-aligned, ENU
                      |
                      |  CORRECTION  -- jumps when the estimator re-localises
                      v
                     odom                      smooth, continuous, drifts
                      |
                      |  ODOMETRY  -- continuous, from the EKF / VIO
                      v
                  base_link                    body frame, FLU, at the CoG
                      |
      +---------------+---------------+-----------------+
      |               |               |                 |
   imu_link      camera_link      lidar_link       gimbal_base
                      |                                 |
              camera_optical_frame               gimbal_camera_link
                (static, REP-103 -> optical)
```

Every edge is a `geometry_msgs/TransformStamped`. Fixed edges (`base_link` ->
sensors) go out once on `/tf_static` with `TRANSIENT_LOCAL` durability. Moving
edges (`map` -> `odom`, `odom` -> `base_link`) go out continuously on `/tf`.

## What each frame is for

| Frame | Fixed? | Meaning |
|---|---|---|
| `earth` | yes | ECEF. Only needed if you run multiple robots with different `map` origins, or care about absolute geodetic pose. Most single-vehicle stacks skip it. |
| `map` | yes | World-fixed, gravity-aligned, x East / y North / z Up. Anchored once at the arming position or a surveyed point. **Discontinuous**: it jumps when GPS re-acquires or a loop closes. |
| `odom` | yes | World-fixed and **continuous**. Never jumps. Drifts without bound. |
| `base_link` | no | The vehicle body. REP-103 FLU: x Forward, y Left, z Up. Origin at the centre of gravity, not at the flight controller. |
| `imu_link` | no | Where the IMU physically is, with its own axis convention. |
| `camera_link` | no | Where the camera body is, in FLU. |
| `camera_optical_frame` | no | Same physical place, rotated to the optical convention: z Forward (out of the lens), x Right, y Down. |
| `lidar_link` | no | Lidar mount point. Check whether your driver publishes in FLU or in its own sensor frame; several do not follow REP-103. |

## Why the split between `map` and `odom` exists

This is the part that gets skipped, and it is the whole point of the tree.

- **`odom` -> `base_link`** is your dead-reckoned pose: IMU integration, visual
  odometry, wheel/airspeed. It is **smooth and continuous** — consecutive
  samples never jump. It also **drifts**, without bound, forever.
- **`map` -> `odom`** is the correction that a global estimator (GPS fusion,
  SLAM loop closure, fiducial localisation) applies. It is **discontinuous** —
  it jumps by whatever the correction is. It does **not** drift.

Splitting them means each consumer picks the property it needs:

- A **controller** integrates `odom` -> `base_link`. It must never see a jump.
  A position setpoint tracker fed a discontinuous pose will command a step, and
  a step into a position controller is a lurch — full stick deflection for one
  cycle. On a real airframe that is a visible pitch-up and a startled safety
  pilot; on an aggressive tune it is a crash.
- A **planner or a geofence** works in `map`. It needs a pose that stays
  correct in the world over an hour, and it can absorb a jump because it
  replans.

If you collapse the two, you have to choose which consumer to break. There is
no third option.

## The shortcut: `static_transform_publisher 0 0 0 0 0 0 odom base_link`

You will see this suggested constantly, and it appears to work. It does not.

Publishing a static identity for `odom` -> `base_link` says **"the vehicle
never moves relative to the odometry frame."** That is a lie the moment the
vehicle leaves the pad, and here is what it actually does:

1. **RViz looks fine and is lying.** Your point cloud renders, the robot model
   appears. Both are drawn at the origin forever. It looks like a working TF
   tree because the lookups succeed.
2. **Every `tf2` transform of sensor data is wrong.** A lidar scan transformed
   into `odom` lands wherever the vehicle was at t=0, not where it is. Feed
   that to a mapper and you get a smeared blob centred on the launch point. The
   mapper does not error; it produces a confidently wrong map.
3. **You cannot add real odometry later.** Two publishers on the same edge is a
   malformed tree. `tf2` will interleave them, and lookups will return the
   identity or the real pose depending on timing. Symptom: a robot that
   teleports between the origin and its true position at a few Hz. The
   `/tf` monitor calls it "TF_REPEATED_DATA", which nobody reads.
4. **It hides the actual problem.** The reason people reach for it is
   `"Could not find a connection between 'map' and 'base_link'"`. That message
   is correct and useful: something that should be publishing odometry is not.
   The fix is to start that thing, not to fake its output.

**What to do instead**

- **You have an EKF (PX4/ArduPilot, robot_localization, VIO).** Publish
  `odom` -> `base_link` from it. For MAVROS, `/mavros/local_position/odom`
  already carries exactly this; either let `mavros` publish TF
  (`local_position/tf/send: true`) or bridge it yourself. Do not do both.
- **You have GPS but no local estimator.** Publish `map` -> `odom` from your
  global fix and `odom` -> `base_link` from IMU integration. Yes, the latter
  drifts. That is the honest representation of what you know.
- **You genuinely have no odometry** — bench testing a sensor, checking a URDF.
  Then publish `map` -> `base_link` directly and leave `odom` out of the tree
  entirely. A short honest tree beats a long dishonest one, and nothing
  downstream can mistake it for real localisation.

## Where MAVROS fits

With `local_position/tf/send: true`, MAVROS publishes:

```
map -> base_link          (from /mavros/local_position/pose)
```

Note there is **no `odom` frame** in that default. MAVROS collapses the two,
because PX4's EKF2 output is already a fused global estimate. That is a
reasonable default for a GPS-flown vehicle where every jump is small. It stops
being reasonable the moment you add a local estimator (VIO, lidar odometry)
that needs a smooth frame of its own — at which point you introduce `odom` and
route MAVROS's pose to `map` -> `odom` instead.

MAVROS publishes ENU/FLU on its ROS topics, not PX4's NED/FRD. It has already
applied the conversion documented in `drone_bringup/core/frames.py`. **Do not
apply it a second time.** Mixing a raw MAVLink stream and a MAVROS topic in the
same node, and converting both, is how you end up with a heading that is 90
degrees off and a roll sign that flips.

## Frame conventions, stated once

| | Convention |
|---|---|
| ROS world (REP-103) | ENU: x East, y North, z Up |
| ROS body (REP-103/105) | FLU: x Forward, y Left, z Up |
| PX4 / MAVLink world | NED: x North, y East, z Down |
| PX4 / MAVLink body | FRD: x Forward, y Right, z Down |
| Camera optical (REP-103) | z Forward, x Right, y Down |
| ROS quaternion order | `(x, y, z, w)` — scalar **last** |
| PX4 quaternion order | `(w, x, y, z)` — scalar **first** |

`drone_bringup/core/frames.py` implements every conversion between these, with
tests that check physical meaning rather than component values. Read the module
docstring before writing your own.

## Checking a tree

```bash
ros2 run tf2_tools view_frames                    # writes frames.pdf
ros2 run tf2_ros tf2_echo map base_link           # live transform
ros2 topic hz /tf                                 # is anything publishing?
ros2 topic echo /tf_static --qos-durability transient_local --once
```

Four things to look for:

1. **One publisher per edge.** Two is a malformed tree, not a redundant one.
2. **No unexplained jumps in `odom` -> `base_link`.** If it jumps, something
   global is being published on the wrong edge.
3. **`/tf_static` is TRANSIENT_LOCAL.** If it is not, a node that starts late
   never receives the static transforms and every lookup through them fails
   with "frame does not exist" — while `view_frames` shows the frame, because
   it started earlier.
4. **Timestamps are sane.** Everything on `/tf` must share a clock. Mixing
   `use_sim_time:=true` and `false` across nodes produces lookups that fail
   with "extrapolation into the future", which reads like a TF bug and is a
   clock bug.
