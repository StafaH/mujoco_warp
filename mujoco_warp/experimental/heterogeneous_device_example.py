"""Change worlds' robots using only GPU episode logic inside a captured graph.

uv run --no-sync python -m mujoco_warp.experimental.heterogeneous_device_example
"""

import json

import mujoco
import warp as wp

from mujoco_warp.experimental import HeterogeneousWorlds

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _episode_logic(
  # In:
  tick: wp.array[int],
  model_ids: wp.array[int],
  # Out:
  requests_out: wp.array[int],
):
  world = wp.tid()
  # Worlds 0 and 1 exchange robot types every four steps; world 2 continues its episode.
  if tick[0] % 4 == 3 and world < 2:
    requests_out[world] = 1 - model_ids[world]


@wp.kernel
def _advance_tick(
  # Data out:
  tick_out: wp.array[int],
):
  tick_out[0] += 1


@wp.kernel
def _route_actions(
  # In:
  world_ids: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
):
  row, actuator = wp.tid()
  world = world_ids[row]
  if world >= 0:
    ctrl_out[row, actuator] = 0.1 * float(world + 1)


def _robot(joints):
  bodies = "".join(
    f'<body pos="{0.5 * i} 0 1"><joint name="j{i}" axis="0 1 0"/>'
    '<geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/></body>'
    for i in range(joints)
  )
  motors = "".join(f'<motor joint="j{i}"/>' for i in range(joints))
  return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody><actuator>{motors}</actuator></mujoco>")


def main():
  worlds = HeterogeneousWorlds([_robot(1), _robot(2)], [1, 0, 1], device="cuda:0")
  requests = worlds.enable_device_resets()
  tick = wp.zeros(1, dtype=int, device=worlds.device)

  def episodes(batch):
    wp.launch(_episode_logic, dim=3, inputs=[tick, batch.world_model_ids], outputs=[requests], device=batch.device)
    wp.launch(_advance_tick, dim=1, outputs=[tick], device=batch.device)

  def actions(batch):
    for group in batch.groups:
      wp.launch(
        _route_actions, dim=group.data.ctrl.shape, inputs=[group.world_ids], outputs=[group.data.ctrl], device=batch.device
      )

  worlds.capture(steps_per_launch=4, before_reset=episodes, before_step=actions)
  graph = worlds.graph
  # No host reset, mapping readback, state copy, allocation, or recapture in this loop.
  for _ in range(4):
    worlds.step()
  assert worlds.graph is graph
  # Readbacks below are final diagnostics only.
  assert worlds.device_reset_status.numpy()[0] == 0
  assert worlds.device_reset_epoch.numpy().tolist() == [4, 4, 0]
  print(
    json.dumps(
      {
        "world_model_ids": worlds.world_model_ids.numpy().tolist(),
        "episodes_reset_per_world": worlds.device_reset_epoch.numpy().tolist(),
        "same_graph": True,
        "memory": worlds.memory_report(),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
