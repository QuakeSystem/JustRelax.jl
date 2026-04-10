# Periodic BC Implementation in LaMEM (for porting to justRelax)

This note summarizes how periodic boundary conditions (PBCs) are implemented in LaMEM and how to port the same design to another staggered-grid finite-difference solver.

The focus is x-periodic topology (`periodic = 1`), because that is what LaMEM supports.

## 1) High-level design

LaMEM implements periodicity as a **grid topology choice**, not as a post-processing copy step:

- Parse one runtime flag: `periodic`.
- Build DMDAs with `DM_BOUNDARY_PERIODIC` in x.
- Adjust staggered-grid object sizes in x to avoid duplicate periodic nodes.
- Use wrapped neighbor indexing in stencils (or PETSc periodic ghosting).
- Skip non-periodic x-side boundary constraints when periodic is active.

This is robust and avoids "manual left-right synchronization" hacks.

## 2) Input and global flag

### User-facing option

`doc/options/input_file.dat`:

- documents `periodic = 1`
- warns periodic topology is only implemented in x direction

### Read flag in code

`src/fdstag.cpp`, `FDSTAGCreate(...)`:

- reads `periodic` into `fs->periodic`
- selects boundary type in x:
  - periodic: `DM_BOUNDARY_PERIODIC`
  - non-periodic: `DM_BOUNDARY_GHOSTED`

## 3) Topology construction (core of implementation)

`src/fdstag.cpp`, `FDSTAGCreateDMDA(...)` is the most important place.

When periodic in x:

- `BC_NONE` and `BC_GHOSTED` become `DM_BOUNDARY_PERIODIC`.
- `bc_node = 1`.
- DMs with x-node-like topology (`DA_COR`, `DA_XY`, `DA_XZ`, `DA_X`) are created with size `Nx - bc_node`.

Why: in periodic x there is no distinct "last node = first node + Lx" DOF; one copy is removed.

This prevents duplicated unknowns at periodic seam.

## 4) Neighbor and halo communication

`src/fdstag.cpp`, `FDSTAGGetNeighbProc(...)`:

- periodic flags (`ptx = fs->periodic`, `pty=0`, `ptz=0`)
- neighbor ranks obtained with `getGlobalRankPeriodic(...)`

So the processor-neighbor map wraps around in x automatically.

## 5) BC compatibility checks

`src/bc.cpp` in BC creation/check stage:

When periodic is active, LaMEM rejects combinations that conflict with periodic topology, e.g.:

- left/right no-slip
- prescribed boundary face velocities (`bvel_face`)
- fixed phase/cell constraints intended for side boundaries
- some strain-period options that imply another periodic mechanism

This is important: periodic topology should not be mixed with contradictory side-wall BC logic.

## 6) Residual and stencil behavior under periodicity

`src/JacRes.cpp`:

- Uses `periodic = fs->periodic`.
- For several edge/corner stencil accesses, x-index clamping at boundaries is only done if `!periodic`.
  - in periodic mode, wrapped topology/ghosting supplies the neighbor values.

Also in copy/constraint routines:

- x-side `SET_TPC(...)` constraints are guarded by `if(!periodic)` for relevant fields.
- Therefore periodic x faces are not treated as physical boundaries with mirrored/Dirichlet ghost constraints.

Equivalent behavior exists in:

- `src/JacResTemp.cpp` (temperature residual/matrix)
- `src/matBFBT.cpp` (pressure preconditioner/operator assembly paths)

## 7) Marker advection periodicity

`src/advect.cpp`:

- periodic marker advection flag is enabled when `fs->periodic` (or specific background-strain periodic mode).
- code enforces compatible advection scheme choices for periodic mode.
- marker position correction/wrapping in periodic x is applied.

So periodicity is consistently used both in Eulerian solve and Lagrangian marker transport.

## 8) Minimal port recipe for justRelax (staggered FD)

Use this order:

1. **Single global flag**
   - `periodic_x = true/false`.

2. **Build periodic topology at grid-object level**
   - Do not duplicate right boundary x-node DOF when periodic.
   - For x-node-like staggered locations, use `Nx-1` if your non-periodic layout uses `Nx`.

3. **Halo exchange with wrap**
   - Left ghost from right interior.
   - Right ghost from left interior.
   - Do this for all staggered arrays that use x-neighbors.

4. **Stencil indexing rule**
   - For index `i`:
     - periodic: use wrap (`im1 = wrap(i-1)`, `ip1 = wrap(i+1)`)
     - non-periodic: clamp or use physical BC ghost value logic.

5. **Disable side boundary constraints in x when periodic**
   - no Dirichlet/Neumann left-right physical wall treatment in x.
   - keep y/z boundary handling unchanged.

6. **Residual/Jacobian consistency**
   - Apply the same periodic wrap logic in:
     - momentum residual
     - continuity residual
     - preconditioner/Jacobian assembly
     - temperature/advection operators

7. **Marker wrapping**
   - after marker position update:
     - if `x < xmin`: `x += Lx`
     - if `x >= xmax`: `x -= Lx`

8. **Validation tests**
   - Uniform flow crossing seam remains continuous.
   - Shear test with periodic x matches non-periodic interior solution away from side walls.
   - Global mass residual unchanged by seam crossing.

## 9) Pseudocode skeleton

```text
if periodic_x:
    nx_node_like = nx_nonperiodic - 1
else:
    nx_node_like = nx_nonperiodic

build_staggered_arrays(nx_node_like, ny, nz, ...)

for each timestep/nonlinear iteration:
    exchange_ghosts_x_with_wrap_if_periodic()

    for each stencil point:
        if periodic_x:
            im1 = wrap(i-1, nx_local_or_global)
            ip1 = wrap(i+1, nx_local_or_global)
        else:
            im1, ip1 = boundary_nonperiodic(i)

        assemble_residual_and_matrix_using(im1, ip1)

    if !periodic_x:
        apply_x_side_physical_BC_constraints()

    apply_yz_boundary_constraints()
```

## 10) Key LaMEM files to inspect while porting

- `src/fdstag.cpp`
  - `FDSTAGCreate`
  - `FDSTAGCreateDMDA`
  - `FDSTAGGetNeighbProc`
- `src/bc.cpp`
  - periodic compatibility checks
- `src/JacRes.cpp`
  - periodic branches in stencil/constraint logic
- `src/JacResTemp.cpp`
- `src/matBFBT.cpp`
- `src/advect.cpp`

## 11) Practical note

If your justRelax code already uses array wrapping utilities, map all x-neighbor fetches to those wrappers and remove separate left/right BC code paths when `periodic_x = true`. The most common bug is leaving one operator (often preconditioner or temperature) non-periodic while momentum/continuity are periodic.

