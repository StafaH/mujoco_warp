#!/usr/bin/env python3
"""Debug script to render a single image using Madrona Warp."""

import argparse

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from etils import epath
from PIL import Image


def _load_model(path: epath.Path) -> mujoco.MjModel:
  if not path.exists():
    resource_path = epath.resource_path("mujoco_warp") / path
    if not resource_path.exists():
      raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
    path = resource_path

  print(f"Loading model from: {path}...")
  if path.suffix == ".mjb":
    return mujoco.MjModel.from_binary_path(path.as_posix())

  spec = mujoco.MjSpec.from_file(path.as_posix())
  # check if the file has any mujoco.sdf test plugins
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjw)
  return spec.compile()


def render_single_image(
    xml_path: str,
    nworld: int = 1,
    world_to_render: int = 0,
):
    """Render a single image and save it."""
    print(f"Loading model from: {xml_path}")

    # Initialize Warp
    wp.init()

    mjm = _load_model(epath.Path(xml_path))
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    print(f"Model loaded: {mjm.ngeom} geometries, {mjm.ncam} cameras")

    # Create Madrona Warp render model and data
    with wp.ScopedDevice(None):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nworld)
        mjw.build_warp_bvh(m, d)

        print(f"Render model created with resolution: {m.render_width}x{m.render_height}")
        print(f"Render data created with shape: {d.pixels.shape}")

        # Render the image
        print("Rendering image...")
        mjw.render(m, d)

        # Get the rendered pixels
        pixels = d.pixels.numpy()
        print(f"Pixels shape: {pixels.shape}")

        # Save the rendered image
        img_array = pixels[world_to_render, 0].reshape(
            m.render_height, m.render_width, 3)
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        image.save("debug.png")

        # Save the depth image
        img_array = d.depth.numpy()
        img_array = img_array[world_to_render, 0] / 5.0
        img_array = img_array.reshape(
            m.render_height, m.render_width)
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        image.save("debug_depth.png")


def main():
    parser = argparse.ArgumentParser(description="Render a single image using Madrona Warp")
    parser.add_argument("--mjcf", help="Path to MuJoCo XML or MJB file")
    parser.add_argument("--nworld", type=int, default=1,
                       help="Number of worlds to simulate")
    parser.add_argument("--world_to_render", type=int, default=0,
                       help="World to render")

    args = parser.parse_args()

    # Check if input file exists
    if not epath.Path(args.mjcf).exists():
        # Try to find it in test_data
        test_path = epath.resource_path("madrona_warp") / "test_data" / args.mjcf
        if test_path.exists():
            args.mjcf = str(test_path)
        else:
            raise FileNotFoundError(f"File not found: {args.mjcf}")

    # Render the image
    render_single_image(
        args.mjcf,
        args.nworld,
        args.world_to_render,
    )


if __name__ == "__main__":
    main()
