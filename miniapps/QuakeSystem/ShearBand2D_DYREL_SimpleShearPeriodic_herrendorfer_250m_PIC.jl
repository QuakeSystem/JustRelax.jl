#=
Herrendörfer-style viscoelastic simple shear with Rate-and-State Friction (DYREL).

- Left/right: periodic velocity BCs (no IGG `periodx`); single MPI rank
- Bottom: no-slip; top: prescribed `V_top` [m/s]
- Geometry: LaMEM-style segmented mesh (uniform x for periodicity; refined y) + GMG fault
- RSF: frozen Ω during DYREL iterations; Ω → Ω_old after converged step
- Inertia: optional LaMEM-style ``ρ (V − V0)/dt`` via `inertia` / `ρ_inertia` toggles
- Step scheme follows LaMEM `-snes_type ksponly`: the RSF law is folded once from the
  incoming velocity, Stokes is solved with that viscosity frozen, and the law is
  re-evaluated at the new velocity before the stress / state are committed.
  `rsf_picard_max > 1` repeats this (LaMEM SNES Picard iterations).
- dt control (LaMEM `tssolve` + `ADVUpdateTimeStepRSF`): grow by `inc_dt`, cap by
  `dt_max` and the rate-and-state limit `dt_rsf`, floor at `dt_min`. No step retries.
- VTK: ParaView fields including Vp_rsf, visc_creep and visc_total (LaMEM names)
- Particles: phases, both Ω locations (vertex `pΩ`, center `pΩc`) and stress live on
  markers. The grid is filled from markers before every step.
  Markers move only after the step is accepted. `displacement_x` is how far each
  marker has traveled, which is what shows the shear in VTK; the horizontal fault
  phase itself does not change shape.
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
# using Pkg; Pkg.activate("miniapps")

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

using GeoParams
using Adapt

include(joinpath(@__DIR__, "simple_shear_setup.jl"))
include(joinpath(@__DIR__, "simple_shear_rheology.jl"))

# Analytic Maxwell τxy for Couette: ε̇ = V_top / ly (engineering shear rate)
solution(ε̇, t, G, η) = ε̇ * η * (1 - exp(-G * t / η))

"""LaMEM `visc_total`: ``1 / (1/η + 1/(G dt))``. `η = Inf` returns `G dt`."""
maxwell_viscosity(η, Gdt) = @. inv(inv(η) + inv(Gdt))

# `dx` is the horizontal displacement since the marker was seeded, folded into
# [-Lx/2, Lx/2] so a periodic crossing does not jump by a full domain width.
@parallel_indices (I...) function _particle_displacement!(dx, px, x0, index, Lx)
    for ip in cellaxes(dx)
        @index(index[ip, I...]) == 0 && continue
        δ = @index(px[ip, I...]) - @index(x0[ip, I...])
        @index dx[ip, I...] = δ - Lx * round(δ / Lx)
    end
    return nothing
end

# One particle phase per marker, from the fault box. `pΩ` / `pΩc` both start at the
# phase's Ω so either RSF location (or both) can be scattered onto the grid.
@parallel_indices (I...) function _init_fault_particles!(
        phases, Ωv, Ωc, x0, px, py, index, y_lo, y_hi, Ω_wall, Ω_fault
    )
    for ip in cellaxes(phases)
        @index(index[ip, I...]) == 0 && continue
        on_fault = (@index(py[ip, I...]) ≥ y_lo) & (@index(py[ip, I...]) ≤ y_hi)
        @index phases[ip, I...] = ifelse(on_fault, 2.0, 1.0)
        Ω = ifelse(on_fault, Ω_fault, Ω_wall)
        @index Ωv[ip, I...] = Ω
        @index Ωc[ip, I...] = Ω
        @index x0[ip, I...] = @index px[ip, I...]
    end
    return nothing
end

function init_fault_particles!(pPhases, pΩ, pΩc, pX0, particles, y_lo, y_hi, Ω_wall, Ω_fault)
    @parallel (@idx size(pPhases)) _init_fault_particles!(
        pPhases, pΩ, pΩc, pX0,
        particles.coords[1], particles.coords[2], particles.index,
        y_lo, y_hi, Ω_wall, Ω_fault,
    )
    return nothing
end

"""Copy marker phase, stress and both Ω fields onto the grid before a solve attempt."""
function sync_particles_to_grid!(stokes, particles, pτ, phase_ratios, pPhases, rsf_bundle, pΩ, pΩc, xci, xvi)
    stress2grid!(stokes, pτ, particles)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    isnothing(rsf_bundle) && return nothing
    fields = rsf_bundle.fields
    if rsf_do_vertex(fields)
        particle2grid!(fields.Ωv, pΩ, particles; ghost_1 = false, ghost_2 = false)
        copyto!(fields.Ω_old_v, fields.Ωv)
    end
    if rsf_do_center(fields)
        particle2centroid!(fields.Ω, pΩc, particles; ghost_1 = false, ghost_2 = false, ghost_3 = false)
        copyto!(fields.Ω_old, fields.Ω)
    end
    # b(x) is an Eulerian patch, so it is rebuilt from the phases now on the grid.
    refresh_rsf_ab_mask!(fields, rsf_bundle.ctrl, phase_ratios, xci, xvi)
    return nothing
end

"""Sample the accepted step onto markers, then move them. Rejected attempts never get here."""
function advect_accepted_particles!(
        stokes, particles, pτ, pPhases, pΩ, pΩc, pX0, pDx, rsf_bundle, material_x, displacement_x, lx, dt
    )
    rotate_stress!(pτ, stokes, particles, dt)
    fields = isnothing(rsf_bundle) ? nothing : rsf_bundle.fields
    if !isnothing(fields)
        rsf_do_vertex(fields) && grid2particle!(pΩ, fields.Ωv, particles; ghost_1 = false, ghost_2 = false)
        rsf_do_center(fields) && centroid2particle!(pΩc, fields.Ω, particles; ghosted = false)
    end
    advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt; periodic_1 = true)
    move_particles!(particles, (pPhases, pΩ, pΩc, pX0, unwrap(pτ)...); periodic_1 = true)
    # `material_x` is each marker's initial x. `displacement_x` is how far it has traveled.
    @parallel (@idx size(pDx)) _particle_displacement!(pDx, particles.coords[1], pX0, particles.index, lx)
    particle2grid!(material_x, pX0, particles; ghost_1 = false, ghost_2 = false)
    particle2grid!(displacement_x, pDx, particles; ghost_1 = false, ghost_2 = false)
    Ω_vertex = isnothing(fields) ? material_x : fields.Ωv
    Ω_center = isnothing(fields) ? material_x : fields.Ω
    inject_particles_phase!(
        particles,
        pPhases,
        (pΩ, pΩc, pX0, unwrap(pτ)...),
        (Ω_vertex, Ω_center, material_x, stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.ω.xy),
    )
    @parallel (@idx size(pDx)) _particle_displacement!(pDx, particles.coords[1], pX0, particles.index, lx)
    particle2grid!(material_x, pX0, particles; ghost_1 = false, ghost_2 = false)
    particle2grid!(displacement_x, pDx, particles; ghost_1 = false, ghost_2 = false)
    return nothing
end

function phase_map(phase_ratios_loc)
    # Adapt bulk-copies CellArray.data; Array(CellArray) scalar-indexes on CUDA
    return [argmax(p) for p in Adapt.adapt(Array, phase_ratios_loc)]
end

"""Pad face residuals onto the cell grid for VTK (trailing faces filled with 0)."""
function residuals_to_centers(stokes)
    nx, ny = size(stokes.P)
    Rx, Ry = Array(stokes.R.Rx), Array(stokes.R.Ry)
    Rx_c, Ry_c = zeros(nx, ny), zeros(nx, ny)
    Rx_c[axes(Rx, 1), axes(Rx, 2)] .= Rx
    Ry_c[axes(Ry, 1), axes(Ry, 2)] .= Ry
    return Rx_c, Ry_c
end

function prepare_vtk!(VTK)
    take(VTK.folder)
    if VTK.do_vtk
        take(VTK.vtk_dir)
        pvd_path = joinpath(VTK.vtk_dir, VTK.pvd_name * ".pvd")
        isfile(pvd_path) && rm(pvd_path)
        for f in readdir(VTK.vtk_dir)
            if startswith(f, "vtk_") || endswith(f, ".pvd")
                rm(joinpath(VTK.vtk_dir, f); force = true)
            end
        end
    end
    return nothing
end

function write_vtk_step!(
        VTK, it, t, xvi, xci, stokes, phase_ratios, Vx_v, Vy_v, zeros_c, τ_analytic, rsf_fields, G, dt,
        material_x, displacement_x,
    )
    VTK.do_vtk || return nothing
    vtk_every = Int(get(VTK, :vtk_every, 1))
    (it == 1 || rem(it, vtk_every) == 0) || return nothing

    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    phase_vertex = phase_map(phase_ratios.vertex)
    phase_center = phase_map(phase_ratios.center)
    τxy_analytic = fill(τ_analytic, size(zeros_c))
    Rx_c, Ry_c = residuals_to_centers(stokes)

    # LaMEM visc_creep is the stored Maxwell buffer. visc_total is what the stress
    # update uses. Point data is the shear viscosity (vertices, where the fault lives
    # for loc = :vertex); cell data is the normal-stress viscosity.
    Gdt = G * dt
    η_c = Array(stokes.viscosity.η)
    η_v = Array(stokes.viscosity.ηv)

    data_v = (;
        Vx = Array(Vx_v),
        Vy = Array(Vy_v),
        phase = phase_vertex,
        material_x = Array(material_x),
        displacement_x = Array(displacement_x),
        visc_creep = η_v,
        visc_total = maxwell_viscosity(η_v, Gdt),
    )
    data_c = (;
        P = Array(stokes.P),
        T = zeros_c,
        phase = phase_center,
        visc_creep = η_c,
        visc_total = maxwell_viscosity(η_c, Gdt),
        τII = Array(stokes.τ.II),
        εII = Array(stokes.ε.II),
        Rx = Rx_c,
        Ry = Ry_c,
        τxy_analytic = τxy_analytic,
    )
    if !isnothing(rsf_fields)
        # Centers (normal stress nodes)
        data_c = merge(
            data_c,
            (;
                Ω = Array(rsf_fields.Ω),
                Vp_rsf = Array(rsf_fields.Vp),
                log10_Vp_rsf = log10.(max.(Array(rsf_fields.Vp), 1.0e-30)),
                τ_rsf = Array(rsf_fields.τ_rsf),
                a_eff = Array(rsf_fields.a_eff),
                b_eff = Array(rsf_fields.b_eff),
            ),
        )
        # Vertices (shear / LaMEM XY-edge nodes) — where τ.xy lives
        Vp_v = Array(rsf_fields.Vp_v)
        data_v = merge(
            data_v,
            (;
                τxy = Array(stokes.τ.xy),
                Ω_v = Array(rsf_fields.Ωv),
                Vp_rsf_v = Vp_v,
                log10_Vp_rsf_v = log10.(max.(Vp_v, 1.0e-30)),
                τ_rsf_v = Array(rsf_fields.τ_rsf_v),
            ),
        )
    else
        data_c = merge(
            data_c,
            (;
                Ω = zeros_c,
                Vp_rsf = zeros_c,
                log10_Vp_rsf = zeros_c,
                τ_rsf = zeros_c,
                a_eff = zeros_c,
                b_eff = zeros_c,
            ),
        )
    end
    velocity_v = (Array(Vx_v), Array(Vy_v))
    path_vtk = joinpath(VTK.vtk_dir, "vtk_" * lpad("$it", 6, "0"))
    save_vtk(
        path_vtk, xvi, xci, data_v, data_c, velocity_v;
        t = t, pvd = joinpath(VTK.vtk_dir, VTK.pvd_name),
    )
    # println("Saved VTK → $(joinpath(VTK.vtk_dir, VTK.pvd_name)).pvd  (it=$it, t=$t)")
    return nothing
end

function main(
        igg;
        # LaMEM-style segments (preferred). Periodic x must stay uniform (one segment).
        nel_x = 64,
        coord_x = (-75.0e3, 75.0e3),
        nel_y = (20, 8, 20),
        coord_y = (-150.0e3, -76.0e3, -74.0e3, 0.0),
        bias_x = nothing,
        bias_y = nothing,
        fault = (; x = (-75.0e3, 75.0e3), y = (-75.5e3, -74.5e3)),
        V_top = 4.0e-9,
        η0 = 5.0e26,
        G0 = 3.0e10,
        ν = 0.25,
        P0 = 5.0e6,
        dt = 500.0,
        nsteps = 5,
        rsf_nt = nothing,
        inertia = false,
        ρ_inertia = nothing,   # nothing → phase-averaged Density field; Number or array override
        VTK = nothing,
    )
    mesh = segmented_grid_2D(;
        nel_x = nel_x, coord_x = coord_x,
        nel_y = nel_y, coord_y = coord_y,
        bias_x = bias_x, bias_y = bias_y,
    )
    (; xvi, ni, li, origin) = mesh
    nx, ny = ni
    lx, ly = li

    isnothing(VTK) && (
        VTK = (;
            do_vtk = true,
            folder = joinpath(@__DIR__, "VTK", "pic"),
            vtk_dir = joinpath(@__DIR__, "VTK", "pic", "vtk"),
            name = "simple_shear_herrendorfer_pic",
            pvd_name = "simple_shear_herrendorfer_pic",
            vtk_every = 1,
        )
    )
    prepare_vtk!(VTK)

    staggered_grid, ph_vertex, _T = simple_shear_2D(xvi, 1250.0, fault, VTK)
    # Nonuniform Geometry on the ParallelStencil backend (CuArray spacings for CUDA kernels)
    grid = Geometry(PTArray(backend), xvi...)
    (; xci, xvi) = grid
    @assert Array(xvi[1])[1] ≈ staggered_grid.xvi[1][1]
    @assert grid.ni == ni

    rheology = init_rheology_simple_shear(; η = η0, G = G0, ν = ν)
    # Markers carry the phase. The GMG mask above is only the reference VTK geometry.
    nxcell, max_xcell, min_xcell = 16, 32, 8
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    pPhases, pΩ, pΩc, pX0, pDx = init_cell_arrays(particles, Val(5))
    pτ = StressParticles(particles)
    phase1 = isnothing(rsf_nt) ? (;) : get(rsf_nt, :Phase1, (;))
    phase2 = isnothing(rsf_nt) ? (;) : get(rsf_nt, :Phase2, (;))
    init_fault_particles!(
        pPhases, pΩ, pΩc, pX0, particles,
        Float64(fault.y[1]), Float64(fault.y[2]),
        Float64(get(phase1, :state_rsf_init, 0.0)),
        Float64(get(phase2, :state_rsf_init, 0.0)),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    println(
        "particles: $(nxcell)/cell (min $(min_xcell), max $(max_xcell)); " *
            "phase and both Ω (vertex pΩ, center pΩc) advect only after an accepted step"
    )

    rsf_enabled = !isnothing(rsf_nt) && Bool(get(rsf_nt, :enabled, true))
    rsf_bundle = nothing
    if rsf_enabled
        loc = get(rsf_nt, :loc, :both)
        # Wf default uses min cell size (fault width floor)
        di_min = ntuple(d -> minimum(Array(grid.di.vertex[d])), 2)
        ctrl = build_rate_state_controller(rsf_nt; nphases = length(rheology), di = di_min)
        fields = RateStateArrays(backend, ni; loc = loc)
        init_rate_state_fields!(fields, ctrl, phase_ratios, xci, xvi)
        rsf_bundle = (; ctrl = ctrl, fields = fields)
    end

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, P0)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # LaMEM eta_min / eta_max: bounds on the creep viscosity, not on the RSF root
    viscosity_cutoff = (1.0e-3, 5.0e26)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # Host coords for comprehensions / VTK (device xvi would scalar-index on CUDA)
    xvi_h = Array.(xvi)
    xci_h = Array.(xci)
    yVx = Array(grid.xi_vel[1][2])
    ybot, ytop = origin[2], origin[2] + ly
    stokes.V.Vx .= PTArray(backend)(
        [
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi_h[1], y in yVx
        ]
    )
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    # Initial marker x, on vertices. Simple shear leaves the horizontal phase unchanged;
    # this field is what moves in VTK.
    material_x = @zeros(ni .+ 1...)
    displacement_x = @zeros(ni .+ 1...)
    particle2grid!(material_x, pX0, particles; ghost_1 = false, ghost_2 = false)
    zeros_c = zeros(nx, ny)

    # grid.di is a NamedTuple of (possibly vector) spacings — DYREL / Gershgorin use @dx/@dy
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5)
    εbg = V_top / ly
    dt_min = rsf_enabled ? Float64(get(rsf_nt, :dt_min, 1.0e-1)) : 0.0
    dt_max = rsf_enabled ? Float64(get(rsf_nt, :dt_max, 1.0e7)) : Inf
    inc_dt = Float64(get(something(rsf_nt, (;)), :inc_dt, 0.1))
    rsf_picard_max = rsf_enabled ? Int(get(rsf_nt, :rsf_picard_max, 1)) : 1
    rsf_picard_rtol = rsf_enabled ? Float64(get(rsf_nt, :rsf_picard_rtol, 1.0e-4)) : 1.0e-4

    println(
        "grid: ni=$ni; Δx∈$(extrema(Array(grid.di.vertex[1]))); " *
            "Δy∈$(extrema(Array(grid.di.vertex[2]))); periodic_x OK (uniform Δx)"
    )

    # Density for inertia: nothing → phase-averaged field; Number → uniform; array → as-is
    ρ_for_inertia = if !inertia
        nothing
    elseif isnothing(ρ_inertia)
        ρ_c = @zeros(ni...)
        compute_ρ!(ρ_c, phase_ratios, rheology, args)
        ρ_c
    elseif ρ_inertia isa Number
        Float64(ρ_inertia)
    else
        ρ_inertia
    end
    if inertia
        ρ_ext = ρ_for_inertia isa Number ?
            (ρ_for_inertia, ρ_for_inertia) :
            extrema(Array(ρ_for_inertia))
        println("inertia ON: ρ ∈ $ρ_ext (phase-averaged field unless overridden)")
        # Start from rest old-velocity so the first step sees ρ V / dt
        fill!(stokes.V0.Vx, 0.0)
        fill!(stokes.V0.Vy, 0.0)
    end

    t, it = 0.0, 0
    t_run = time()
    rsf_enabled && println(
        "RSF: LaMEM Picard scheme, rsf_picard_max = $rsf_picard_max (1 = ksponly), rsf_picard_rtol = $rsf_picard_rtol; dt ∈ [$dt_min, $dt_max], inc_dt = $inc_dt"
    )
    for _ in 1:nsteps
        t_step = time()
        # Stress / state / phases come from the markers (not moved since the last accepted step)
        sync_particles_to_grid!(
            stokes, particles, pτ, phase_ratios, pPhases, rsf_bundle, pΩ, pΩc, xci, xvi
        )
        out = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 50.0e4,
                nout = 10,
                rel_drop = 1.0e-6,
                linear_viscosity = true,
                viscosity_cutoff = viscosity_cutoff,
                rsf = rsf_bundle,
                inertia = inertia,
                ρ_inertia = ρ_for_inertia,
                rsf_picard_max = rsf_picard_max,
                rsf_picard_rtol = rsf_picard_rtol,
            )
        )
        # Keep Dirichlet top wall under inertia updates
        @views stokes.V.Vx[:, end] .= V_top
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)

        # Markers move only once the step above has been accepted.
        advect_accepted_particles!(
            stokes, particles, pτ, pPhases, pΩ, pΩc, pX0, pDx, rsf_bundle, material_x, displacement_x, lx, dt
        )
        update_phase_ratios!(phase_ratios, particles, pPhases)

        it += 1
        t += dt

        τxy_max = maximum(abs, stokes.τ.xy)
        τ_analytic = solution(εbg, t, G0, η0)
        println("it = $it; t = $t; max|τxy| = $τxy_max; analytic = $τ_analytic; dt = $dt; dt_rsf = $(out.dt_rsf); itPH = $(out.itPH); picard = $(out.rsf_picard_its); ||F||/||F0|| = $(out.rsf_residual)")
        println("max |displacement_x| = $(maximum(abs, Array(displacement_x)))")
        println("Timestep completed in $(round(time() - t_step; digits = 2)) sec, made $(out.iter) iterations.")

        write_vtk_step!(
            VTK, it, t, xvi_h, xci_h, stokes, phase_ratios, Vx_v, Vy_v, zeros_c, τ_analytic,
            isnothing(rsf_bundle) ? nothing : rsf_bundle.fields,
            G0, dt, material_x, displacement_x,
        )

        # LaMEM: dt grows by inc_dt, capped by dt_max and the rate-and-state limit dt_rsf
        # (ADVUpdateTimeStepRSF), floored at dt_min. The step just taken is never redone.
        if rsf_enabled
            dt_next = min(dt * (1 + inc_dt), dt_max)
            isfinite(out.dt_rsf) && (dt_next = min(dt_next, out.dt_rsf))
            dt = max(dt_next, dt_min)
            args = (; args..., dt = dt)
        end
    end

    println("Total runtime: $(round(time() - t_run; digits = 2)) sec")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    # -----------------------------------------------------------------------------
    # Default run — LaMEM-style segments (Herrendörfer / Shear_test_PBC)
    # Periodic x: single uniform segment. Refined y around the fault band.
    nel_x = 600
    coord_x = (-75.0e3, 75.0e3)
    nel_y = (10, 4, 10)                          # cells per y-segment (LaMEM nel_z)
    coord_y = (-150.0e3, -75.5e3, -74.5e3, 0.0)  # segment boundaries (LaMEM coord_z)
    fault = (; x = (-75.0e3, 75.0e3), y = (-75.2e3, -74.8e3))
    V_top = 4.0e-9
    dt = 500.0  # LaMEM initial dt; grows by inc_dt up to dt_max
    nsteps = 103000
    # Physical inertia ρ(V−V0)/dt (LaMEM `inertia = 1`). Off by default — set true for inertial runs.
    inertia = true
    ρ_inertia = nothing  # nothing → phase-averaged Density field (both phases 2700 here)
    nx, ny = sum(nel_x isa Integer ? (nel_x,) : Tuple(nel_x)), sum(nel_y isa Integer ? (nel_y,) : Tuple(nel_y))

    RSF = (
        enabled = true,
        affect_stokes = true,
        # Both locations: vertex shear (pΩ) and cell centers (pΩc).
        loc = :vertex,
        V0 = 4.0e-9,
        # LaMEM Shear_test_PBC_herrendorfer.dat: dt_min = 1e-1, dt_max = 1e7, inc_dt = 0.1
        dt_min = 1.0e-1,
        dt_max = 1.0e7,
        inc_dt = 0.1,
        # 1 = LaMEM `-snes_type ksponly`; > 1 = Picard iterations of the frozen-η solve
        rsf_picard_max = 10,
        rsf_picard_rtol = 1.0e-4,
        G = 3.0e10,
        ν = 0.25,
        # Both phases carry a_rsf (LaMEM / Herrendörfer Shear_test_PBC)
        Phase1 = (
            a_rsf = 0.011,
            b_rsf = 0.017,
            mu0_rsf = 0.2,
            D_rs = 0.01,
            Wf = 250.0,
            state_rsf_init = 40.0,
            λ = 0.0,
            C = 0.0,
        ),
        Phase2 = (
            a_rsf = 0.011,
            b_rsf = 0.001,
            b_rsf_val = (0.001, 0.017, 0.017, 0.001),
            b_rsf_x = (-47000.0, -43000.0, 33000.0, 37000.0),
            mu0_rsf = 0.2,
            D_rs = 0.01,
            Wf = 250.0,
            state_rsf_init = -1.0,
            λ = 0.0,
            C = 0.0,
        ),
    )

    VTK = (;
        do_vtk = true,
        folder = joinpath(@__DIR__, "VTK", "pic"),
        vtk_dir = joinpath(@__DIR__, "VTK", "pic", "vtk"),
        name = "simple_shear_herrendorfer_pic",
        pvd_name = "simple_shear_herrendorfer_pic",
        vtk_every = 1,
    )

    igg = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
    else
        igg
    end

    @time main(
        igg;
        nel_x = nel_x,
        coord_x = coord_x,
        nel_y = nel_y,
        coord_y = coord_y,
        fault = fault,
        V_top = V_top,
        dt = dt,
        nsteps = nsteps,
        rsf_nt = RSF,
        inertia = inertia,
        ρ_inertia = ρ_inertia,
        VTK = VTK,
    )
end
