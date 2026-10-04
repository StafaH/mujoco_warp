"""Graph-compatible snapshots of native MJWarp contacts in a DynamicTable.

MJWarp already packs contacts across each model batch into a global ``naconmax``
buffer. This adapter adds storage and copies for downstream inspection/export;
it does not replace collision allocation or provide contacts to the native solver.
Requires the experimental ``warp.experimental.table`` prototype.
"""

import warp as wp
from warp.experimental.table import DynamicTable
from warp.experimental.table import TableState
from warp.experimental.table import append_row

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _count_contacts(
  # Data in:
  nacon_in: wp.array[int],
  ncollision_in: wp.array[int],
  # In:
  source_world: wp.array[int],
  world_ids: wp.array[int],
  native_capacity: int,
  width: int,
  # Out:
  state_out: TableState,
  errors_out: wp.array[int],
  counts_out: wp.array[int],
):
  if wp.tid() == 0 and (nacon_in[0] > native_capacity or ncollision_in[0] > native_capacity):
    wp.atomic_or(errors_out, 0, 1)
  for source in range(wp.tid(), wp.min(nacon_in[0], native_capacity), width):
    slot = source_world[source]
    if slot < 0 or slot >= world_ids.shape[0]:
      wp.atomic_or(errors_out, 0, 2)
    else:
      world = world_ids[slot]
      if world != -1:
        if world < 0 or world >= state_out.num_worlds:
          wp.atomic_or(state_out.counters, 2, 2)
        else:
          wp.atomic_add(counts_out, world, 1)


@wp.kernel
def _scatter_contacts(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  source_world: wp.array[int],
  source_position: wp.array[wp.vec3],
  source_frame: wp.array[wp.mat33],
  source_dist: wp.array[float],
  source_geom: wp.array[wp.vec2i],
  world_ids: wp.array[int],
  native_capacity: int,
  width: int,
  offsets: wp.array[int],
  # Out:
  cursors_out: wp.array[int],
  position_out: wp.array[wp.vec3],
  normal_out: wp.array[wp.vec3],
  depth_out: wp.array[float],
  geom1_out: wp.array[int],
  geom2_out: wp.array[int],
):
  for source in range(wp.tid(), wp.min(nacon_in[0], native_capacity), width):
    slot = source_world[source]
    if slot >= 0 and slot < world_ids.shape[0]:
      world = world_ids[slot]
      if world >= 0 and world < cursors_out.shape[0]:
        if offsets[world] >= 0:
          row = offsets[world] + wp.atomic_add(cursors_out, world, 1)
          position_out[row] = source_position[source]
          normal_out[row] = source_frame[source][0]
          depth_out[row] = -source_dist[source]
          geom1_out[row] = source_geom[source][0]
          geom2_out[row] = source_geom[source][1]


@wp.kernel
def _snapshot_contacts(
  # Data in:
  nacon_in: wp.array[int],
  ncollision_in: wp.array[int],
  # In:
  source_world: wp.array[int],
  source_position: wp.array[wp.vec3],
  source_frame: wp.array[wp.mat33],
  source_dist: wp.array[float],
  source_geom: wp.array[wp.vec2i],
  world_ids: wp.array[int],
  native_capacity: int,
  width: int,
  # Out:
  state_out: TableState,
  errors_out: wp.array[int],
  position_out: wp.array[wp.vec3],
  normal_out: wp.array[wp.vec3],
  depth_out: wp.array[float],
  geom1_out: wp.array[int],
  geom2_out: wp.array[int],
):
  if wp.tid() == 0 and (nacon_in[0] > native_capacity or ncollision_in[0] > native_capacity):
    wp.atomic_or(errors_out, 0, 1)
  for source in range(wp.tid(), wp.min(nacon_in[0], native_capacity), width):
    slot = source_world[source]
    if slot < 0 or slot >= world_ids.shape[0]:
      wp.atomic_or(errors_out, 0, 2)
    else:
      world = world_ids[slot]
      # Unassigned physical slots have no logical world to export.
      if world != -1:
        row = append_row(state_out, world)
        if row >= 0:
          position_out[row] = source_position[source]
          normal_out[row] = source_frame[source][0]
          depth_out[row] = -source_dist[source]
          geom1_out[row] = source_geom[source][0]
          geom2_out[row] = source_geom[source][1]


class ContactTable:
  """Snapshot and group contacts by logical world using device counts and mappings.

  Call ``rebuild(groups)`` after collision detection/stepping. Each group supplies
  ``data`` and a device ``world_ids`` array mapping physical slots to logical IDs
  (``-1`` skips an inactive slot). Geometry IDs remain local to the source model.
  ``depth`` is signed penetration, i.e. minus native distance, and ``normal`` points
  from geom1 to geom2. Margin contacts may have negative depth.

  Rebuild clears, appends, and compacts inside a captured graph without host reads.
  Commit capacity with ``table.ensure_capacity`` between completed replays. Growth
  never occurs during rebuild; check sticky errors outside capture. Partial snapshots
  after overflow must not be interpreted as complete. Rows within a world are unordered.
  """

  def __init__(self, num_worlds, max_rows, initial_capacity=0, device=None, dispatch_width=4096):
    """Reserve snapshot columns and commit the requested initial row capacity."""
    self.table = DynamicTable(
      {"position": wp.vec3, "normal": wp.vec3, "depth": float, "geom1": int, "geom2": int},
      max_rows=max_rows,
      num_worlds=num_worlds,
      initial_capacity=initial_capacity,
      device=device,
      dispatch_width=dispatch_width,
    )
    self.errors = wp.zeros(1, dtype=int, device=self.table.device)
    self._counts = wp.zeros(num_worlds, dtype=int, device=self.table.device)
    self._cursors = wp.zeros(num_worlds, dtype=int, device=self.table.device)

  @property
  def workspace_bytes(self):
    """Snapshot count/cursor/error allocation bytes outside DynamicTable storage."""
    return self.errors.capacity + self._counts.capacity + self._cursors.capacity

  @property
  def columns(self):
    """Reserved component arrays; use world_offsets to delimit live rows."""
    return self.table.columns

  @property
  def world_offsets(self):
    """Device offsets delimiting each logical world's contacts after rebuild."""
    return self.table.world_offsets

  def append(self, data, world_ids):
    """Append one batch's native contacts; recordable inside a CUDA graph."""
    if data.nacon.device != self.table.device:
      raise ValueError("contact data must be on the table device")
    if world_ids.device != self.table.device or world_ids.dtype != wp.int32 or world_ids.shape != (data.nworld,):
      raise ValueError("world_ids must be an int32 array of nworld elements on the table device")
    columns = self.columns
    contact = data.contact
    wp.launch(
      _snapshot_contacts,
      dim=self.table.dispatch_width,
      inputs=[
        data.nacon,
        data.ncollision,
        contact.worldid,
        contact.pos,
        contact.frame,
        contact.dist,
        contact.geom,
        world_ids,
        data.naconmax,
        self.table.dispatch_width,
      ],
      outputs=[self.table.state, self.errors, *[columns[name] for name in ("position", "normal", "depth", "geom1", "geom2")]],
      device=self.table.device,
    )

  def rebuild(self, groups):
    """Count, bulk allocate, and scatter contacts before publishing world offsets.

    One batch reservation avoids serializing all contacts on a global append
    counter. A capacity failure leaves an empty snapshot and sticky error.
    """
    groups = tuple(groups)
    for group in groups:
      if group.data.nacon.device != self.table.device:
        raise ValueError("contact data must be on the table device")
      mapping = group.world_ids
      if mapping.device != self.table.device or mapping.dtype != wp.int32 or mapping.shape != (group.data.nworld,):
        raise ValueError("world_ids must be an int32 array of nworld elements on the table device")
    self.table.clear()
    self._counts.zero_()
    self._cursors.zero_()
    for group in groups:
      data = group.data
      wp.launch(
        _count_contacts,
        dim=self.table.dispatch_width,
        inputs=[data.nacon, data.ncollision, data.contact.worldid, group.world_ids, data.naconmax, self.table.dispatch_width],
        outputs=[self.table.state, self.errors, self._counts],
        device=self.table.device,
      )
    offsets = self.table.append_worlds(self._counts)
    columns = self.columns
    for group in groups:
      data = group.data
      contact = data.contact
      wp.launch(
        _scatter_contacts,
        dim=self.table.dispatch_width,
        inputs=[
          data.nacon,
          contact.worldid,
          contact.pos,
          contact.frame,
          contact.dist,
          contact.geom,
          group.world_ids,
          data.naconmax,
          self.table.dispatch_width,
          offsets,
        ],
        outputs=[self._cursors, *[columns[name] for name in ("position", "normal", "depth", "geom1", "geom2")]],
        device=self.table.device,
      )
    self.table.compact()

  def clear_errors(self):
    """Reset sticky native-source and table overflow errors on the device."""
    self.errors.zero_()
    self.table.clear_errors()

  def check_errors(self):
    """Synchronize and raise if the native source or snapshot is incomplete."""
    errors = int(self.errors.numpy()[0])
    if errors & 1:
      raise RuntimeError("Native MJWarp contact/pair capacity exhausted; increase naconmax and recompute contacts")
    if errors & 2:
      raise ValueError("Native contact has an invalid physical world ID")
    self.table.check_errors()
