"""GPU episode logic changes the entire robot population under one graph.

uv run --no-sync python -m mujoco_warp.experimental.pooled_example
"""

import json

import numpy as np
import warp as wp

from mujoco_warp.experimental import PooledHeterogeneousWorlds
from mujoco_warp.experimental.heterogeneous_device_example import _robot

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _episodes(
  # In:
  tick: wp.array[int],
  # Out:
  requests_out: wp.array[int],
):
  world = wp.tid()
  if tick[0] == 8:
    requests_out[world] = 1  # Every world becomes a two-joint robot.
  elif tick[0] == 16:
    requests_out[world] = 2  # Every world becomes a three-joint robot.
  elif tick[0] == 24:
    requests_out[world] = world % 3  # Return to a heterogeneous population.


@wp.kernel
def _advance_tick(
  # Data out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


@wp.kernel
def _actions(
  # Data out:
  ctrl_out: wp.array2d[float],
):
  world, actuator = wp.tid()
  ctrl_out[world, actuator] = 0.1 * float(world + 1)


def main():
  worlds = PooledHeterogeneousWorlds(
    [_robot(1), _robot(2), _robot(3)],
    np.arange(6, dtype=np.int32) % 3,
    batch_size=2,
    scratch_bytes=8 << 20,
    device="cuda:0",
  )
  tick = wp.zeros(1, dtype=int, device=worlds.device)

  def episodes(batch):
    wp.launch(_episodes, dim=batch.nworld, inputs=[tick], outputs=[batch.reset_model_ids], device=batch.device)
    wp.launch(_advance_tick, dim=1, outputs=[tick], device=batch.device)

  def actions(batch):
    wp.launch(_actions, dim=batch.state.ctrl.shape, outputs=[batch.state.ctrl], device=batch.device)

  worlds.capture(steps_per_launch=8, before_reset=episodes, before_step=actions)
  graph, pointer = worlds.graph, worlds.state.qpos.ptr
  for _ in range(4):
    worlds.step()
  # Inspection happens only after the complete GPU-driven rollout.
  assert worlds.graph is graph and worlds.state.qpos.ptr == pointer
  np.testing.assert_array_equal(worlds.device_reset_epoch.numpy(), 3)
  assert not np.any(worlds.device_reset_status.numpy())
  assert not np.any(worlds.physics_overflow.numpy())
  assert not np.any(worlds.physics_nonfinite.numpy())
  print(
    json.dumps(
      {
        "final_model_ids": worlds.world_model_ids.numpy().tolist(),
        "resets_per_world": worlds.device_reset_epoch.numpy().tolist(),
        "same_graph_and_state_pointer": True,
        "memory": worlds.memory_report(),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
