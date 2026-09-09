# ros2_controller — template controller that owns `/mjpc_bridge`

The ROS 2 half of the link. `templates/ros2_bridge/` is the mjpc half; the protocol and
its failure modes are in `MJPC_ROS2_BRIDGE_GUIDE.md`.

This exists because the real controller (`franka_ec::MPPITrackController`) lives outside
this repository, so a clone gives you the planner and no way to reach a robot. This
package is a complete, buildable stand-in.

**Verified**: `g++ -fsyntax-only -std=c++17` against ROS 2 Humble headers, clean.
Never run on hardware — treat it as reviewed source, not tested behaviour.

```
package.xml  CMakeLists.txt                     ament package
example_mjpc_controller_plugin.xml              pluginlib export (see below)
include/example_mjpc_controller/
    example_mjpc_controller.hpp
    mjpc_bridge.h                               canonical 120 B struct
src/example_mjpc_controller.cpp                 the whole protocol, ~190 lines
config/controllers.yaml                         controller_manager entry
launch/example_mjpc_controller.launch.py        spawner only
```

## Build

```bash
cp -r templates/ros2_controller ~/your_ws/src/example_mjpc_controller
cd ~/your_ws && colcon build --packages-select example_mjpc_controller
source install/setup.bash
```

Merge `config/controllers.yaml` into whatever `controller_manager` config your robot
bringup already loads, bring the hardware up first, then spawn:

```bash
ros2 launch <your_robot>_bringup <hardware>.launch.py robot_ip:=172.16.0.2
ros2 launch example_mjpc_controller example_mjpc_controller.launch.py
```

Confirm it took, before involving mjpc:

```bash
cd .../templates/ros2_bridge
g++ -O2 -o bridge_probe bridge_probe.cc -lrt
./bridge_probe watch     # state_seq must RISE
./bridge_probe zero      # no "action timeout" in the controller log
```

## What differs from franka_ec's real controller

| | real | template |
|---|---|---|
| state interfaces | franka `robot_state` + `robot_model` (FK) plus joint pos/vel | joint pos/vel only |
| `MjpcBridge::ee_pos` | never written | never written |
| unused gain members | present | removed |
| protocol, LPF, rate limit, ceiling, timeout, ownership | — | **identical** |

Dropping FK is safe and was checked, not assumed: **nothing writes `ee_pos` and no mjpc
task reads it.** Those 12 bytes exist only to keep the struct layout fixed. That also
means the template runs on any 7-DOF arm exposing effort command interfaces, not just
an FR3.

## The four things that are easy to get wrong

**1. Effort, not position.** `action` is N·m. A position-command controller would read
newton-metres as radians — ±87 rad of commanded angle — and nothing warns you. The
sibling `mppi_pos_controller` in franka_ec is the one that wants setpoints.

**2. `pluginlib_export_plugin_description_file`.** Without that CMake line the library
builds and installs and the class is simply never found; the error says the controller
type does not exist, which does not point at CMake.

**3. `kTauMax` is FR3-specific** — `{87,87,87,87,12,12,12}`. Change it for another arm,
or the ceiling either never trips or trips constantly.

**4. Zero effort means gravity compensation *on this hardware interface*,** because the
robot's own controller holds `g(q)`. All three fallbacks command zero. On an arm where
zero effort means the arm falls, the fallback has to become an explicit gravity torque
before this is safe to run.

## Do not simplify these

- **Ownership.** The controller creates on activate and **unlinks** on deactivate; mjpc
  only ever attaches. A leftover region would be read as live by the next mjpc run.
- **Counter last.** Write the payload, then bump the sequence counter. Reversed, the
  reader can take a half-written sample.
- **New actions only.** `action_seq > last_action_seq_`, never "read every cycle".
  Reusing a stale action is how a dead planner becomes a moving arm; counting staleness
  is what makes the timeout possible at all.
- **Latching fallbacks.** Both set `action_ready_ = false`, so recovery needs a fresh
  `action_seq` rather than the offending condition merely going away.
