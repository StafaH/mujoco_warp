# Copyright 2026 The Newton Developers
# Licensed under the Apache License, Version 2.0.
"""Renderer compatibility and dynamic-scene tests for the optional OptiX backend."""

import functools
from unittest import mock

import mujoco
import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import render_optix
from mujoco_warp._src import render_test


def _require_optix():
  if not wp.is_cuda_available():
    pytest.skip("OptiX requires CUDA")
  try:
    render_optix._paths()
  except RuntimeError as error:
    pytest.skip(str(error))


class OptixRenderTest(render_test.RenderTest):
  """Exercise the existing image-rendering tests through OptiX."""

  def setUp(self):
    super().setUp()
    _require_optix()
    self.enterContext(
      mock.patch.object(mjw, "create_render_context", functools.partial(mjw.create_render_context, backend="optix"))
    )


def test_backend_validation():
  mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
  with pytest.raises(ValueError, match="Unknown rendering backend"):
    mjw.create_render_context(mjm, backend="typo")
  with wp.ScopedDevice("cpu"), pytest.raises(ValueError, match="requires an NVIDIA CUDA device"):
    mjw.create_render_context(mjm, backend="optix")


def _compare(left, right):
  np.testing.assert_allclose(left.depth_data.numpy(), right.depth_data.numpy(), atol=1e-4, rtol=1e-5)
  np.testing.assert_array_equal(left.seg_data.numpy(), right.seg_data.numpy())
  a, b = left.rgb_data.numpy(), right.rgb_data.numpy()
  for shift in (0, 8, 16):
    np.testing.assert_allclose(((a >> shift) & 255).astype(int), ((b >> shift) & 255).astype(int), atol=2, rtol=0)


@pytest.mark.parametrize("shadows,samples,nworld", [(False, 1, 3), (True, 2, 17)])
def test_batched_dynamic_mesh_instances(shadows, samples, nworld):
  _require_optix()
  xml = """
  <mujoco>
    <asset>
      <mesh name="a" vertex="-.5 -.5 0  .5 -.5 0  0 .5 0  0 0 1"/>
      <mesh name="b" vertex="-.3 -.3 0  .3 -.3 0  0 .3 0  0 0 .6"/>
    </asset>
    <worldbody>
      <camera pos="0 -4 2" xyaxes="1 0 0 0 .4 1"/>
      <light pos="-1 -2 4"/>
      <geom type="plane" size="0 0 .1"/>
      <geom type="mesh" mesh="a" rgba=".8 .4 .2 1"/>
      <geom type="capsule" pos="1 0 .6" size=".2 .4"/>
    </worldbody>
  </mujoco>
  """
  mjm, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
  contexts = [
    mjw.create_render_context(
      mjm,
      nworld=nworld,
      backend=backend,
      cam_res=(48, 48),
      render_rgb=True,
      render_depth=True,
      render_seg=True,
      use_shadows=shadows,
      samples_per_pixel=samples,
      use_precomputed_rays=samples > 1,
    )
    for backend in ("warp", "optix")
  ]
  for rc in contexts:
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
  _compare(*contexts)
  ids = np.tile(mjm.geom_dataid, (nworld, 1))
  ids[1::2, 1] = 1
  m.geom_dataid = wp.array(ids, dtype=int)
  positions = d.geom_xpos.numpy()
  positions[:, 1, 0] += np.linspace(-0.5, 0.5, nworld)
  d.geom_xpos.assign(positions)
  rotations = d.geom_xmat.numpy()
  for world, angle in enumerate(np.linspace(-0.4, 0.7, nworld)):
    c, s = np.cos(angle), np.sin(angle)
    rotations[world, 1] = rotations[world, 1] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  d.geom_xmat.assign(rotations)
  # More than eight worlds exercises reuse of visibility-mask bits.
  # Replaying transform packing must feed current data to the hardware scene build.
  with wp.ScopedCapture() as capture:
    mjw.refit_bvh(m, d, contexts[1])
  for _ in range(3):
    positions[:, 1, 0] += np.linspace(-0.03, 0.03, nworld)
    d.geom_xpos.assign(positions)
    for rc in contexts:
      if rc.backend == "optix":
        wp.capture_launch(capture.graph)
      else:
        mjw.refit_bvh(m, d, rc)
      mjw.render(m, d, rc)
    _compare(*contexts)
  depth = contexts[1].depth_data.numpy()
  assert not np.array_equal(depth[0], depth[1])
  assert not np.array_equal(depth[1], depth[2])


@pytest.mark.parametrize("backend", ["warp", "optix"])
def test_empty_scene_and_no_active_cameras(backend):
  if backend == "optix":
    _require_optix()
  mjm, _, m, d = test_data.fixture(xml="<mujoco><worldbody><camera/></worldbody></mujoco>", nworld=2)
  for active in (None, []):
    rc = mjw.create_render_context(
      mjm, nworld=2, cam_active=active, backend=backend, cam_res=(8, 8), render_depth=True, render_seg=True
    )
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    assert not np.any(rc.depth_data.numpy())
    assert np.all(rc.seg_data.numpy() == -1)


def test_capture_is_rejected_without_invalidating_cuda():
  _require_optix()
  mjm, _, m, d = test_data.fixture(
    xml='<mujoco><asset><mesh name="m" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>'
    '<worldbody><camera/><geom type="mesh" mesh="m"/></worldbody></mujoco>'
  )
  rc = mjw.create_render_context(mjm, backend="optix", cam_res=(8, 8), render_depth=True)
  mjw.render(m, d, rc)
  with wp.ScopedCapture():
    with pytest.raises(RuntimeError, match="cannot be captured"):
      mjw.render(m, d, rc)
  mjw.render(m, d, rc)
  wp.synchronize()


@pytest.mark.parametrize("path", ["flex/rope.xml", "flex/floppy.xml"])
def test_deforming_flex(path):
  _require_optix()
  mjm, _, m, d = test_data.fixture(path=path, nworld=2)
  contexts = [
    mjw.create_render_context(
      mjm, nworld=2, backend=backend, cam_res=(32, 32), render_rgb=True, render_depth=True, render_seg=True
    )
    for backend in ("warp", "optix")
  ]
  if mjm.nflex > 1:
    # Separate the two ropes: their touching endpoints otherwise have ambiguous IDs.
    vertices = d.flexvert_xpos.numpy()
    vertices[:, mjm.flex_vertadr[1] :, 0] += 0.5
    d.flexvert_xpos.assign(vertices)
  for offset in (0.0, 0.1):
    vertices = d.flexvert_xpos.numpy()
    vertices[1, :, 0] += offset
    d.flexvert_xpos.assign(vertices)
    for rc in contexts:
      mjw.refit_bvh(m, d, rc)
      mjw.render(m, d, rc)
    _compare(*contexts)


def test_hfield_on_nondefault_stream():
  _require_optix()
  xml = """
  <mujoco>
    <asset><hfield name="hill" size="1 1 .3 .1" nrow="3" ncol="3" elevation="0 0 0 0 1 0 0 0 0"/></asset>
    <worldbody>
      <camera pos="0 -3 2" xyaxes="1 0 0 0 .7 1"/>
      <geom type="hfield" hfield="hill"/>
    </worldbody>
  </mujoco>
  """
  mjm, _, m, d = test_data.fixture(xml=xml, nworld=2)
  wp.synchronize()
  with wp.ScopedStream(wp.Stream()):
    contexts = [
      mjw.create_render_context(
        mjm, nworld=2, backend=backend, cam_res=(32, 32), render_rgb=True, render_depth=True, render_seg=True
      )
      for backend in ("warp", "optix")
    ]
    for rc in contexts:
      mjw.refit_bvh(m, d, rc)
      mjw.render(m, d, rc)
    _compare(*contexts)
    assert np.count_nonzero(contexts[1].depth_data.numpy()) > 100
