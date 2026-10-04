"""Run growing device command rows against heterogeneous physics under one CUDA graph.

Run with the locally built Warp fork:
  uv run --no-sync python -m mujoco_warp.experimental.heterogeneous_example

This storage example grows action rows between graph replays. For GPU-generated robot
changes entirely inside the graph, run heterogeneous_device_example instead.
"""

import json

import mujoco
import numpy as np
import warp as wp
from warp.experimental.table import DynamicTable
from warp.experimental.table import TableState
from warp.experimental.table import append_row

from mujoco_warp.experimental import HeterogeneousWorlds

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _append_commands(
  # In:
  state: TableState,
  first_world: int,
  # Out:
  actions_out: wp.array[float],
):
  world = first_world + wp.tid()
  row = append_row(state, world)
  if row >= 0:
    actions_out[row] = 0.1 * float(world + 1)


@wp.kernel
def _apply_commands(
  # In:
  state: TableState,
  actions: wp.array[float],
  world_group: wp.array[int],
  world_row: wp.array[int],
  group_id: int,
  width: int,
  # Data out:
  ctrl_out: wp.array2d[float],
):
  for command in range(wp.tid(), state.counters[0], width):
    if state.alive[command] != 0:
      world = state.world[command]
      if world_group[world] == group_id:
        for actuator in range(ctrl_out.shape[1]):
          ctrl_out[world_row[world], actuator] = actions[command]


def _robot(joints):
  bodies = "".join(
    f'<body pos="{0.5 * i} 0 1"><joint name="j{i}" axis="0 1 0"/>'
    '<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/></body>'
    for i in range(joints)
  )
  motors = "".join(f'<motor joint="j{i}"/>' for i in range(joints))
  return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody><actuator>{motors}</actuator></mujoco>")


def main():
  worlds = HeterogeneousWorlds([_robot(1), _robot(2)], [1, 0, 1], device="cuda:0", memory="virtual")
  commands = DynamicTable({"action": float}, max_rows=1 << 20, num_worlds=3, initial_capacity=1, device="cuda:0")
  action_pointer = commands.columns["action"].ptr
  physics_pointers = [group.data.qpos.ptr for group in worlds.groups]
  wp.launch(_append_commands, dim=1, inputs=[commands.state, 0, commands.columns["action"]], device=worlds.device)

  def before_step(batch):
    for group_id, group in enumerate(batch.groups):
      wp.launch(
        _apply_commands,
        dim=64,
        inputs=[commands.state, commands.columns["action"], batch.world_group, batch.world_row, group_id, 64, group.data.ctrl],
        device=batch.device,
      )

  worlds.capture(before_step=before_step)
  graph = worlds.graph
  worlds.step()
  np.testing.assert_allclose(worlds.world_array(0, "ctrl").numpy(), 0.1)
  np.testing.assert_allclose(worlds.world_array(1, "ctrl").numpy(), 0.0)

  # Grow the table's accessible prefix between graph replays, then add commands on device.
  commands.ensure_capacity(3)
  wp.launch(_append_commands, dim=2, inputs=[commands.state, 1, commands.columns["action"]], device=worlds.device)
  commands.check_errors()
  worlds.step()
  assert worlds.graph is graph
  assert commands.columns["action"].ptr == action_pointer
  assert [group.data.qpos.ptr for group in worlds.groups] == physics_pointers
  for world in range(3):
    np.testing.assert_allclose(worlds.world_array(world, "ctrl").numpy(), 0.1 * (world + 1))

  print(
    json.dumps(
      {
        "live_command_rows": commands.live_count,
        "command_capacity": commands.capacity,
        "command_committed_bytes": commands.committed_bytes,
        "physics": worlds.memory_report(),
        "graph_reused_after_append": True,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
