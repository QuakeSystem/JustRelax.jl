# Rate-and-state friction in JustRelax

Notes from the SEAS BP1-FD and Herrendörfer eigenstrain work. Read this before
changing either rate-and-state path, the DYREL timestep, or the fault mesh.

Two implementations exist and they must not be combined. The error is
`rsf_eigenstrain and rsf with affect_stokes both update the fault stress; pass only one`.

| Path | Where | What it does |
| --- | --- | --- |
| Viscosity fold | `src/rheology/RateState*.jl`, `rsf` with `affect_stokes=true` | LaMEM cell solve. Inside every DYREL iteration, `rsf_effective_viscosity` bisects `η` so `τ = 2 η DII` matches elasticity, Maxwell creep, and the sinh law at frozen `Ω`. |
| Eigenstrain | `src/rheology/rsf_eigenstrain/` | The inner DYREL solve stays elastic. Slip is an eigenstrain on **one vertex row** of `τ.xy`. The sinh law is inverted only after that solve. |

Inertia stays on. Do not check these setups with a quasi-static solve. Do not add
radiation damping `η_rad = G / (2 c_s)` on top of DYREL inertia: the inertia term
`ρ (V − V0) / dt` is already the wave mechanism, and the damper would count it twice.
The scalar slider below uses radiation damping because it has no inertia.

## Names

`Wf` is the LaMEM fault **width**, the thickness in `V / (2 Wf)`. Herrendörfer uses
500 m. SEAS BP1-FD uses 100 m. Both are `Wf`.

`Wf_height` is the SEAS down-dip **height** of the rate-and-state section. The SCEC
papers also call that length `Wf`. In BP1-FD it is 40 km. It is the mask cutoff,
not the fine-grid depth and not the 500 m thickness.

## Eigenstrain step

Source: `law.jl` (pure functions, root `JustRelax` module), `kernels.jl` and
`step.jl` (included from `common.jl`, so every backend compiles them).

- Fault mask: `0` off the fault, `1` rate-and-state (invert and age), `2` prescribed
  slip. BP1 uses mask 2 for `x <= -Wf_height`. Mask 2 keeps `V_lag`, and aging,
  `eigenstrain_lamem_dt`, and `set_eigenstrain_ψ_from_stress!` skip it.
- Trial traction, with `h` the accepted `τ_o` from the previous step:

  `τ = h + 2 G Δt (ε − sign(τ_el) V_lag / (2 Wf))`

  Clamped so it does not cross zero and does not exceed the elastic kernel traction.
- Normal stress in the slip law is the imposed effective stress `σ` (`λ = 0`), not
  the dynamic pressure.
- Non-finite or non-positive rates become 0. `V` is capped at 1000 m/s.
- `eigenstrain_slip_rate` uses `abs(τ)`. A negative traction must be treated as
  locked **before** that call. Otherwise the absolute value creates a second
  high-rate root and any bracket or bisection is invalid.
- Each pass restores the `τ_o` snapshot and, when inertia is on, the full `V0`.
  `solve_DYREL!` copies `τ → τ_o` and `V → V0` at the end, so a pass that is
  rejected would otherwise poison the next one. Age `ψ` once, after the step is
  accepted.
- Public allocator is `eigenstrain_rsf(ni; ...)` or `eigenstrain_rsf(backend, ni; ...)`.
  Do not add a constructor named `EigenstrainRSF`. Julia 1.12 overwrites it during
  precompile because the 2D and 3D modules both include `common.jl`. The struct
  itself stays `EigenstrainRSF`.
- From a device module, call `_solve_DYREL!` and `_eigenstrain_physical_step!`.
  Do not call `solve_DYREL!(stokes)` again: that re-enters the trait dispatch.
  `flatten_solver_kwargs` is defined later in `common.jl`; calling it at runtime
  is fine.

GPU: the bracket is an `@inline` scalar function called from the existing
`@parallel_indices` invert kernel. No allocation, no scalar indexing of device
arrays. `dVdτ` is cleared with `fill!`, which is the same call the variational
solver already runs on CUDA. QuakeSystem scripts keep `isCUDA = false`; switch
that flag only for a device run. Value types and pure functions live in root
`JustRelax` (included before the CPU and GPU modules). Anything that uses
`@zeros` or `@parallel` lives in a file included from `common.jl`.

## The failure: Picard on the slip rate

A frozen-stress update `V ← law(τ(V_lag))` diverges during rupture.

The local eigenstrain slope is `∂τ/∂V ≈ −G Δt / Wf`. For a large argument the
sinh law has `dV/dτ ≈ V / (a σ)`. One fixed-point pass therefore amplifies the
rate error by

```
(V / (a σ)) * (G Δt / Wf)
```

The pass is stable only while that factor is below 1, i.e.

```
Δt < a σ Wf / (G V)
```

With the SEAS numbers `a = 0.01`, `σ = 50 MPa`, `Wf = 100 m`, `G = 3.2038e10`,

```
a σ Wf / G = 1.56e-3  s · (m/s)
```

At `V = 0.04 m/s` and `Δt = 1 s` the gain is about 26. The iteration oscillates:
an even pass stores a huge rate and the next elastic solve clamps `τ` to 0; the
odd pass sees a locked fault and reloads the stress. If `max_pass` is even, the
step can commit `V ≈ 0` and `τ = 0`. A huge `V_lag` inside DYREL also blows the
momentum residual, the velocity goes non-finite, and the inner loop hits
`isnan(err)` (`solver.jl`, "NaN detected in inner loop"). More Picard passes make
this worse, not better. Eight passes on the scalar slider increased the residual.

The LaMEM weakening step sits on the same boundary. `dt_weakening = dteta * L / V`
with `dteta ≈ 0.2` gives a gain

```
(G / Wf) * 0.2 * L / (a σ) ≈ 1.03
```

for these SEAS numbers, **independent of V**. So honoring the weakening timestep
does not by itself make Picard stable. The timestep is chosen from the old rate,
then the rate jumps inside the step and that same `Δt` is unstable.

### Fix

`eigenstrain_consistent_rate` brackets `V` so that

```
law(τ_star − (G Δt / Wf) (V − V_lag)) = V
```

`τ_star` is the traction the elastic solve returned at `V_lag`. A non-positive
extrapolated traction returns rate 0. `Δt = 0` makes the slope zero and recovers
the raw law, which is why `invert_rsf_eigenstrain!(ez, stokes)` with no timestep
still means "invert the frozen stress". The bracket is logarithmic on
`[1e-30, V_cap]` with 80 bisections. A linear bracket on `[0, V_cap]` cannot
resolve an interseismic rate near `1e-9`–`1e-12` m/s.

`G / Wf` is an upper bound on `|∂τ/∂V|`. The domain spring `G / H` is softer, and
once `dy = Wf` the velocity jump cancels part of the eigenstrain. The bracket
therefore under-predicts `V`, and the next elastic pass raises it. That is stable
under-relaxation. The old Picard update was not.

`max_pass` defaults to 8. The 4×40 strip needed 6 passes at the onset of slip to
get the relative rate change under `1e-2`, then 1–2 passes once the slip was
fast. The loop returns as soon as the tolerance is met, so a locked step does
not pay for 8 elastic solves. Stopping at 4 during the onset commits a slightly
low rate; because the bracket under-relaxes, that is still stable, but the rate
lags the fault.

## DYREL pseudo-time state

`α`, `β` and `dτ` are rebuilt at the start of every solve from the current `Δt`:

- viscoelastic viscosity `η_ve = η / (1 + η / (G Δt))`
- Gershgorin `λmax` from that viscosity
- with inertia, `ρ / Δt` is added to the diagonal and `λmax` is updated
- `dτ = 2 CFL / √λmax`, then `α = (2 − c dτ) / (2 + c dτ)`, `β = 2 dτ / (2 + c dτ)`

A fast change of timestep does not leave those coefficients on the old scale.

What does carry over is the pseudo-velocity `dVdτ` and the residual history `R0`.
The update is `dVdτ ← α dVdτ + residual` with `α` near 1, so most of the previous
search direction is kept. After an eigenstrain pass changes the fault traction,
or after the viscosity-fold path drops `η` by orders of magnitude, that kick is
sized for the old operator. The variational solver (`solver_VS.jl`) already
zeros `dVdτ`, `dV`, `R0` and `cV` on every call, because the free-surface mask
can change. Both rate-and-state paths now zero `dVdτ` and `R0` once, at the
start of `_solve_DYREL!`. They accumulate again inside the solve. Zeroing them
on every inner iteration would throw away the acceleration that makes DYREL
converge.

Gershgorin is refreshed from the current viscosity only every `nout` inner
iterations (the miniapps use `nout = 50` or `100`). For up to that many
iterations after a sudden viscosity drop, `dτ` can still be the step of the
locked fault. The viscosity path's cell bisection is already the consistent
local solve; the carried `dVdτ` was the piece that carried over from the
eigenstrain failure. The variational formulation is the same DYREL iteration
on a rock-ratio grid, not a more advanced solver.

## Timestep

`compute_dt_ratestate_lamem` / `eigenstrain_lamem_dt` is the minimum of healing,
weakening and the slip Courant time. `eigenstrain_lamem_dt` looks only at mask
`1`. Mask `2` is omitted there, but `eigenstrain_dt_limit` and
`eigenstrain_slip_change` use `mask > 0.5`, so they **include** the prescribed
plate-rate nodes.

```
k            = (2 / 3.14) * G / ((1 − ν) Wf)     # Wf is the fault width
dt_healing   = 0.2 * L / (V0 * exp(−ψ))
dt_weakening = min(dteta * L / V, 1e9)           # dteta from dteta_max_lamem, else 0.2
dt_courant   = 1e-3 * Wf / V
```

`dteta_max_lamem` clamps `dteta` to `[0.1, 0.2]`. An uncapped healing step on a
locked, well-aged fault is `1e8`–`1e9` s and loads tens of MPa in one step. With
`ΔV ~ V` and `Δt = 0.2 L / V`, the eigenstrain stress change per step is about
`(G / Wf) * 0.2 * L ≈ 0.51 MPa` for the SEAS width. A floor of 1 s is about
25 times too long at `V = 0.04 m/s` and is what wiped the first strip.

The scalar slider adds a loading cap so a locked step gains at most `0.05 a σ`:

```
dt_load = 0.05 * a * σ / (k * Vp)     # here k = G / H, the domain spring
```

At `H = 4 km` that cap is `3.12e6` s. The production BP1 and Herrendörfer drivers
use the LaMEM minimum, not this cap.

Printed `Vmax` in the BP1 log is `0.2 L / eigenstrain_dt_limit`. Because that
limit includes mask 2, `Vmax` stays at the plate rate `1e-9` until some
rate-and-state node exceeds it. A station at 12.5 km can still show `V ~ 1e-14`
while another node on the row is already running away. The 80 km NaN was that
second node.

## Mesh and boundaries

`εxy = 0.5 (∂Vx/∂y + ∂Vy/∂x)`. Across one fault-normal cell of size `dy` with a
plate-rate jump, `εxy = 0.5 Vp / dy`. The eigenstrain subtracts `V / (2 Wf)`.
They cancel only when `dy = Wf`. A 2.5 km cell with `Wf = 100 m` does not
cancel, and it does not resolve nucleation.

Nucleation length for BP1, `(2/π) G / (1−ν) * L / (σ (b−a))`, is a few hundred
metres to about 1 km. 100 m cells resolve it. 2.5 km cells do not.

Critical spring `k_crit = σ (b − a) / L`. SEAS values give `k_crit = 3.125e7 Pa/m`.
`G / Wf = 3.20e8`, so `(G / Wf) / k_crit = 10.25`: a fault that is only one `Wf`
cell thick is stable and only creeps. The domain spring is `k = G / H`. Stick-slip
needs `H ≳ 1 km` (`G / k_crit ≈ 1.0 km`). The lightweight strip uses `H = 4 km`,
`k / k_crit = 0.256`, uniform `a = 0.010`, periodic along strike, `nx = 4`,
`ny = 40`, cells of 100 m.

`cell_sizes_ratio` keeps the fine-to-coarse jump under 3.5 and requires the final
adjacent ratio, including the junction, to stay under 3.6. A larger jump makes
the DYREL step unstable once a vertical velocity appears.

Periodic `y` was rejected for BP1. DYREL periodic directions require uniform
spacing (`check_periodic_bcs`), and a periodic pair copies the tangential
velocity from one side onto the other, so the two plates cannot move at opposite
rates. Sides are driven: bottom wall 0, top ghost `Vx[i, end] = 2 Vp − Vx[i, end−1]`,
because the wall velocity is the average of the ghost and the first interior node.
Both boundary columns of `Vx` are prescribed and are not updated by the momentum
kernel.

The Earth's free surface in the BP1 miniapp is the **right** face, `x = 0`
(`x` is depth, left is deep; `y` is fault-normal). 2D `free_slip` on the left
and right faces only mirrors `Vy`. It does not enforce `∂Vx/∂x = 0`. That
condition is the solver keyword `boundary_vx` on `_solve_DYREL!`:

```julia
boundary_vx = (; free_surface = :right, top = Vp)
```

`:left` and `:right` copy the interior `Vx` column onto the boundary column.
`top` and `bot` are wall velocities. It is applied once before the
Powell–Hestenes loop and after every velocity update, before `update_halo!`.
2D only; a 3D call with a tuple errors. Do not turn on `free_surface = true`
for this problem: that flag is the top `y` face and sets `τyy = P`.

A frozen right `Vx` column kept the plate-rate step across `dy = Wf` and loaded
the shallow fault at `G Vp / Wf ≈ 0.32 Pa/s`. The rupture then looked like it
started at the surface. With the Neumann column, nucleation is at depth.

BP1 coordinates: slip is `Vx`, eigenstrain is the in-plane `τ.xy` row. This is
not a 3D antiplane model. The antiplane free-surface condition
`σ_xz = μ ∂u/∂z = 0` is exactly `∂Vx/∂x = 0` at `x = 0`.

## Initial conditions, do not mix them

Herrendörfer, locked:

```
ψ = φ = −1,   τ = 0,   V ≈ 0
```

Steady sliding at the plate rate is the failed state of that experiment. Stress
has to build under the boundary shear before the velocity-weakening patch fails.

BP1-FD, prestressed. Uniform `τ0` from `a_max`, `V = Vp`, and `ψ(x)` so the
local `a(x)` returns that same stress and rate:

```
τ0 = σ * a_max * asinh( (V / (2 V0)) * exp( (μ0 + b ln(V0 / V)) / a_max ) )
ψ  = ( a * (ln(2 V0 / V) + ln sinh(τ / (a σ))) − μ0 ) / b
```

With `a_max = 0.025`, `σ = 50 MPa`, `μ0 = 0.6`, `b = 0.015`, `V0 = 1e-6`,
`V = 1e-9`: `τ0 = 26.54612236050892 MPa`. The shallow `a = 0.01` steady state
at the same rate is higher, `τ_ss = 31.727 MPa`, `ψ_ss = 6.908`. `a` knots are
`a_rsf_z = (−18000, −15000)`, `a_rsf_val = (0.025, 0.010)`.

LaMEM `state_rsf` (CSV column 22) is `log10(θ)`, not `Ω`. With `D = 0.008` and
`V0 = 1e-6`, `log10(L / V0) = 3.90309`. Comparison stress is `dev_stress_xy`
(column 8). `Vp_rsf` is column 24. Time is column 1, in seconds. ParaView time
is seconds. Log lines may use years; `history.csv` has both.

## What the runs did

Reference series:
`/Users/Ibrag003/project/rsf/LaMEM_3/LaMEM_RSF/examples/BuiltInSetups/domain_search/baseline_series`
(`fault_0km.csv`, `fault_10km.csv`, `fault_12.5km.csv`, `fault_20km.csv`). The
reference event ends near `t = 5.667e9` s, about 179.6 yr.

- Wide domain, `±1000 km` by 74 km, ratio 2, 100 m fault cells, after the
  free-surface fix: killed at about 4 yr. Surface excess was only `+0.06 MPa`,
  12.5 km was `+0.11 MPa`. Output is `VTK_SEAS_BP1_FD`. Relaunching that script
  deletes those files.
- 80×80 km, ratio 3, **2.5 km** fault cells: finished, never ruptured. At
  180 yr the 12.5 km stress was 32.53 MPa against a reference that had already
  dropped to 26.8 MPa. Stations at 0, 10 and 12.5 km were stuck together.
  Stiffness is `~ G/L` of the small box, and the cells neither match `Wf` nor
  resolve nucleation.
- 80×80 km, 20 cells of `Wf = 100 m` across the fault, but still 2.5 km along
  depth: NaN at 97.93 yr, step 1821, inside `_solve_DYREL!` on the **next**
  step. Last complete line had 12.5 km at 31.33 MPa (reference 29.16), station
  `V` still `~6e-14`, while the row's `Vmax` had climbed and `Δt` had collapsed
  from 608 s to 0.3 s. That was the Picard blow-up. Output is
  `VTK_SEAS_BP1_FD_80km`. This run was on the old update.
- 160×160 km, the production mesh: depth `−160 km` to `0`, fault-normal `±80 km`.
  100 m × 100 m around the fault: 20 cells across (2 km, fault on a vertex) and
  100 m from the surface down to 20 km. Ratio 3 toward the sides and below
  20 km. `ni = (207, 32)`, fault row `j = 17`, `dy = (100, 100)`. `Wf_height`
  stays 40 km. That cutoff does **not** land on a vertex: the nearest vertices
  are about `−66.6 km` (mask 2) and `−35.5 km` (mask 1, still inverted). The run
  was stopped from outside at step 368, `t = 19.4 yr`, still loading, 12.5 km
  `τ = 27.42 MPa` against reference 27.10 (`Δ = +0.32 MPa`), plate rate, no NaN.
  It was still the old Picard update. Output is `VTK_SEAS_BP1_FD_160km`.
  `prepare_eigen_vtk!` deletes `step_*`, `.pvd`, `.vtr` and `.csv` in the
  directory it is pointed at. Do not relaunch this script if those files should
  be kept.

Stick-slip probe, same SEAS velocity-weakening numbers, `Δψ = 2`, `H = 4 km`:

- Scalar spring `k = G/H` plus radiation damping, bisection on the rate residual:
  one closed cycle, drop 5.653 MPa, peak 0.979 m/s, back to `V ≤ 1e-9`. Without
  the damper every event hit the 1000 m/s cap. Test:
  `test/test_eigenstrain_stickslip.jl`.
- DYREL strip `4 × 40`, real LaMEM timestep (`dt_floor = 1e-6`, no scalar
  predictor), after the consistent-rate bracket: one event, no NaN. Peak
  1.19 m/s. Stress from 32.73 MPa down to 20.92 MPa (drop 11.8 MPa), then the
  rate fell back below `1e-9` and the stress settled near 21.7 MPa. Residual
  stayed under 1%. The larger drop than the scalar cycle is the inertial strip
  with reflecting walls, not a second wipe. History:
  `VTK_SEAS_stickslip/history.csv`.

## Coseismic NaN: the free-surface column had no inertia

The 160 km GPU run died inside the first elastic solve of the step after
`it = 2270` (`t = 151.19 yr`, `Vmax = 0.109 m/s`, `dt = 1.57e-2 s`, 12.5 km
still at 34.6 MPa) with `NaN detected in inner loop`. The previous step had
converged. The uniform 4×40 strip survives the same `dt`, stress and rate.

`add_inertia_diagonal!` was launched on `size(Dx)`. On a non-periodic grid `Dx`
is `(nx − 1, ny)` and `Dy` is `(nx, ny − 1)`, so the last Vy column was never
given `ρ/dt`. That column is the free-surface row. The residual still subtracts
`ρ(V − V0)/dt` and divides by the viscous diagonal. The 4×40 strip is periodic
in x, so `Dx` is long enough to cover every Vy row. Interseismic steps have
`ρ/dt ~ 10⁻³`, which is nothing next to the viscous diagonal.

At the crashing step `dt ≈ 0.016 s` and `ρ/dt ≈ 1.7e5`. A square 100 m cell has
a viscous diagonal of about `4e5`, so missing the mass is a 40% error and still
inside the CFL margin. On the 160 km mesh that same skipped column is 100 m in
x but tens of kilometres in y, and the diagonal falls to `η_ve / dx² ≈ 5e4`.
`ρ/dt` is then several times the diagonal, `dτ` is past the CFL limit, and the
iteration overflows. The inner-loop check only trips on NaN, so the residual
goes through Inf first, `λmin` becomes `Inf/Inf`, and the next check reports
the NaN.

The launch now uses the same covering range as the pseudo-time update, so every
momentum row gets the mass. The regression is
`test/test_eigenstrain_coseismic_graded.jl`: ratio-3 mesh, free surface on the
right, coarse cell on that skipped column, and the GPU step's `τ`, `V` and `dt`.
It must stay finite, and `min(Dy)` on the free-surface column must be at least
`ρ/dt`.

## Files

- `SEAS_BP1_FD_eigenstrain.jl` — 160 km BP1 eigenstrain driver. `boundary_vx = (; free_surface = :right, top = Vp)`, `max_pass = 8`.
- `ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer_eigenstrain.jl` — Herrendörfer eigenstrain, locked initial state, `Wf = 500 m`.
- `ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer.jl` — viscosity-fold Herrendörfer. Same friction update as before; its DYREL solves now also start from a cleared pseudo-velocity.
- `seas_stickslip_slider.jl`, `SEAS_stickslip_eigenstrain.jl` — the scalar cycle and the 4×40 strip.
- `test/test_dyrel_eigenstrain_rsf_2D.jl` — law, consistent-rate bracket, one inertial DYREL step. `invert_rsf_eigenstrain!` is called there with no `Δt`; keep that default working.
- `test/test_eigenstrain_coseismic_graded.jl` — graded free-surface column at the GPU blow-up step. Every momentum diagonal must carry `ρ/dt`.
- `test/test_eigenstrain_stickslip.jl` — the scalar cycle.

CPU check:

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no test/test_dyrel_eigenstrain_rsf_2D.jl
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no test/test_eigenstrain_coseismic_graded.jl
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no test/test_eigenstrain_stickslip.jl
```
