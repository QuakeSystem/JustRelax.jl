#=
Lightweight stick-slip probe for the eigenstrain rate-and-state split.

The 80 km BP1 run went non-finite in DYREL once a fault node ran away above the
plate rate. This case uses the SEAS BP1-FD velocity-weakening numbers
(`a = 0.010`, `b = 0.015`) on a uniform fault, so nothing varies along strike.

`G / Wf` for `Wf = 100 m` is about 10 times the critical stiffness, so a mesh
that is only one fault cell thick creeps. The fault-normal domain here is
`H = 4 km` (`k = G / H ≈ 0.26 k_crit`), filled with 100 m cells. Along the fault
the grid is 4 cells and periodic.

1. A scalar slider with that same spring (`seas_stickslip_slider.jl`). It has to
   complete stick-slip cycles.
2. The same initial state in DYREL, inertia on, top wall at the plate rate.
   If the fixed-point slip rate misses the tolerance, the step is restored and
   `dt` is halved. That is the stability question the big run hit.

The timestep is the LaMEM rate-and-state step, capped so a locked fault gains at
most `0.05 a σ` per step.

    julia --project=miniapps --startup-file=no miniapps/QuakeSystem/SEAS_stickslip_eigenstrain.jl

`SEAS_SLIDER_DYREL=0` runs only the scalar slider.
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Printf

include(joinpath(@__DIR__, "seas_stickslip_slider.jl"))

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

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

const YEAR = 365.25 * 86400.0

_mean(x) = sum(x) / length(x)

"LaMEM step on the fault row, with the same locked-loading cap as the scalar slider."
function capped_fault_dt(ez, p)
    return clamp(min(eigenstrain_lamem_dt(ez; ν = p.ν), p.dt_load), 1.0e-6, 1.0e9)
end

function init_host_rheology(; η, G, ν, ρ)
    el = ConstantElasticity(; G = G, ν = ν)
    return (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η), el)),
            Elasticity = el,
        ),
    )
end

function apply_plate_velocity!(stokes, Vp)
    Vx = stokes.V.Vx
    @views Vx[:, end] .= 2 * Vp .- Vx[:, end - 1]
    return nothing
end

"Uniform grid. `x` is periodic and along the fault, `y` is fault-normal, fault at 0."
function stickslip_vertices(; nx = 4, ny = 4, h = 100.0)
    iseven(ny) || error("ny must be even so the fault sits on a vertex")
    xv = collect(range(0.0, nx * h; length = nx + 1))
    yv = collect(range(-ny * h / 2, ny * h / 2; length = ny + 1))
    return xv, yv
end

function _host(a)
    a === nothing && return nothing
    return Array(a)
end

function _restore!(dst, src)
    src === nothing && return nothing
    copyto!(dst, src)
    return nothing
end

"Fields the physical step overwrites. `dVdτ` is included so a rejected step can be repeated."
function snapshot_step(stokes, ez, dyrel)
    return (;
        τxx = _host(stokes.τ.xx), τyy = _host(stokes.τ.yy), τxy = _host(stokes.τ.xy),
        τoxx = _host(stokes.τ_o.xx), τoyy = _host(stokes.τ_o.yy), τoxy = _host(stokes.τ_o.xy),
        P = _host(stokes.P),
        Vx = _host(stokes.V.Vx), Vy = _host(stokes.V.Vy),
        V0x = _host(stokes.V0.Vx), V0y = _host(stokes.V0.Vy),
        ψ = _host(ez.ψ), ψ_old = _host(ez.ψ_old), V = _host(ez.V), V_lag = _host(ez.V_lag),
        dVxdτ = _host(dyrel.dVxdτ), dVydτ = _host(dyrel.dVydτ),
    )
end

function restore_step!(stokes, ez, dyrel, snap)
    _restore!(stokes.τ.xx, snap.τxx)
    _restore!(stokes.τ.yy, snap.τyy)
    _restore!(stokes.τ.xy, snap.τxy)
    _restore!(stokes.τ_o.xx, snap.τoxx)
    _restore!(stokes.τ_o.yy, snap.τoyy)
    _restore!(stokes.τ_o.xy, snap.τoxy)
    _restore!(stokes.P, snap.P)
    _restore!(stokes.V.Vx, snap.Vx)
    _restore!(stokes.V.Vy, snap.Vy)
    _restore!(stokes.V0.Vx, snap.V0x)
    _restore!(stokes.V0.Vy, snap.V0y)
    _restore!(ez.ψ, snap.ψ)
    _restore!(ez.ψ_old, snap.ψ_old)
    _restore!(ez.V, snap.V)
    _restore!(ez.V_lag, snap.V_lag)
    _restore!(dyrel.dVxdτ, snap.dVxdτ)
    _restore!(dyrel.dVydτ, snap.dVydτ)
    return nothing
end

function run_dyrel_stickslip(p; nx = 4, nsteps = 2_000, max_events = 1, adaptive = true, dt_floor = 1.0)
    h = p.Wf
    ny = round(Int, p.H / h)
    ny * h == p.H || error("domain height $(p.H) m is not a multiple of the cell size $h m")
    xv, yv = stickslip_vertices(; nx = nx, ny = ny, h = h)
    ni = (nx, ny)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    grid = Geometry(PTArray(backend), xv, yv)
    rheology = init_host_rheology(; η = p.η, G = p.G, ν = p.ν, ρ = p.ρ)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    ez = eigenstrain_rsf(
        backend, ni; Wf = p.Wf, G = p.G, σ = p.σ, law = p.proto,
        V_cap = p.V_cap, tol = p.tol, max_pass = p.max_pass,
    )
    j_fault = init_eigenstrain_row!(
        ez, grid.xvi; y_fault = 0.0, a = p.a, b = p.b, ψ = p.ψ, V_lag = p.V,
    )
    yv_h = Array(grid.xvi[2])
    dy_lo = yv_h[j_fault] - yv_h[j_fault - 1]
    dy_hi = yv_h[j_fault + 1] - yv_h[j_fault]
    (dy_lo ≈ p.Wf && dy_hi ≈ p.Wf) ||
        error("fault cells are ($dy_lo, $dy_hi) m, not Wf = $(p.Wf) m")

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, p.σ)
    fill!(stokes.τ.xy, p.τ)
    fill!(stokes.τ_o.xy, p.τ)
    yVx = Array(grid.xi_vel[1][2])
    stokes.V.Vx .= PTArray(backend)([y >= 0.0 ? p.Vp : 0.0 for _ in xv, y in yVx])
    fill!(stokes.V.Vy, 0.0)
    apply_plate_velocity!(stokes, p.Vp)
    flow_bcs!(stokes, flow_bcs)
    apply_plate_velocity!(stokes, p.Vp)
    stokes.V0.Vx .= stokes.V.Vx
    stokes.V0.Vy .= stokes.V.Vy
    update_halo!(@velocity(stokes)...)

    dt = capped_fault_dt(ez, p)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6, CFL = 0.9)

    outdir = joinpath(@__DIR__, "VTK_SEAS_stickslip")
    mkpath(outdir)
    hist = open(joinpath(outdir, "history.csv"), "w")
    println(hist, "it,t_year,t_s,dt,passes,rel,cuts,tau_MPa,V,psi")

    println(
        "DYREL strip  ni=$(ni)  h=$(h) m  fault j=$j_fault  dy=($dy_lo, $dy_hi)  " *
            "adaptive=$adaptive",
    )
    println("initial dt = $dt s   V = $(p.V) m/s   τ = $(p.τ / 1.0e6) MPa")
    flush(stdout)

    t = 0.0
    events = 0
    phase = :locked
    τ_peak = p.τ
    τ_min = p.τ
    record = NamedTuple[]
    solve_kwargs = (;
        verbose_PH = false,
        verbose_DR = false,
        iterMax = 2.0e4,
        iterMax_PH = 6,
        nout = 50,
        rel_drop = 1.0e-3,
        linear_viscosity = true,
        update_material = false,
        boundary_vx = (; top = p.Vp),
    )
    try
        for it in 1:nsteps
            dt_try = max(capped_fault_dt(ez, p), dt_floor)
            # The scalar slider says when this dt would more than double the rate.
            if adaptive
                τ_now = _mean(Array(stokes.τ.xy)[:, j_fault])
                ψ_now = _mean(Array(ez.ψ)[:, j_fault])
                V_now = maximum(Array(ez.V)[:, j_fault])
                for _ in 1:24
                    dt_try <= dt_floor && break
                    pred = slider_step(p, τ_now, ψ_now, V_now, dt_try)
                    step_is_resolved(p, V_now, pred) && break
                    dt_try = max(dt_try * 0.5, dt_floor)
                end
            end
            snap = adaptive ? snapshot_step(stokes, ez, dyrel) : nothing
            cuts = 0
            out = nothing
            while true
                args = (; args..., dt = dt_try)
                try
                    out = eigenstrain_physical_step!(
                        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid,
                        dt_try, igg, ez;
                        inertia = true, ρ_inertia = p.ρ, solve_kwargs = solve_kwargs,
                    )
                catch err
                    @printf("DYREL failed at it=%d t=%.6f yr dt=%.3e\n", it, t / YEAR, dt_try)
                    println("fault V = ", Array(ez.V)[:, j_fault])
                    println("fault τ = ", Array(stokes.τ.xy)[:, j_fault])
                    rethrow(err)
                end
                V_new = maximum(Array(ez.V))
                finite_step = isfinite(out.rel) && all(isfinite, Array(stokes.τ.xy)) && isfinite(V_new)
                finite_step || error("non-finite DYREL state at it=$it t=$t dt=$dt_try")
                V_old = adaptive ? maximum(snap.V) : 0.0
                resolved = out.rel < p.tol && V_new < 0.5 * p.V_cap &&
                    V_new <= max(V_old, 1.0e-30) * 2
                if !adaptive || resolved || dt_try <= dt_floor || cuts >= 2
                    break
                end
                restore_step!(stokes, ez, dyrel, snap)
                dt_try = max(dt_try * 0.5, dt_floor)
                cuts += 1
            end
            apply_plate_velocity!(stokes, p.Vp)
            t += dt_try
            τ_row = Array(stokes.τ.xy)[:, j_fault]
            V_row = Array(ez.V)[:, j_fault]
            ψ_row = Array(ez.ψ)[:, j_fault]
            τm = _mean(τ_row)
            Vm = maximum(V_row)
            if phase === :locked
                τ_peak = max(τ_peak, τm)
                if Vm >= 1.0e-6
                    phase = :slipping
                    τ_min = τm
                end
            else
                τ_peak = max(τ_peak, τm)
                τ_min = min(τ_min, τm)
                if Vm <= 1.0e-9
                    phase = :locked
                    events += 1
                    @printf(
                        "event %d closed  t=%8.4f yr  drop=%6.3f MPa\n",
                        events, t / YEAR, (τ_peak - τ_min) / 1.0e6,
                    )
                    τ_peak = τm
                    events >= max_events && break
                end
            end
            interesting = it <= 3 || it % 25 == 0 || Vm > 1.0e-8 || cuts > 0 || out.rel > p.tol
            if interesting
                @printf(
                    "it=%d  t=%8.4f yr  dt=%.3e  cuts=%d  passes=%d  rel=%.2e  τ=%7.3f MPa  V=%.3e  ψ=%.3f\n",
                    it, t / YEAR, dt_try, cuts, out.passes, out.rel,
                    τm / 1.0e6, Vm, _mean(ψ_row),
                )
                flush(stdout)
            end
            println(hist, "$it,$(t / YEAR),$t,$dt_try,$(out.passes),$(out.rel),$cuts,$τm,$Vm,$(_mean(ψ_row))")
            push!(record, (; it, t, dt = dt_try, τ = τm, V = Vm, rel = out.rel, cuts))
        end
    finally
        close(hist)
    end
    println("DYREL stopped at t=$(t / YEAR) yr after $(length(record)) steps, events=$events")
    println("history: $(joinpath(outdir, "history.csv"))")
    flush(stdout)
    return record
end

function report_slider(p, hist)
    summary = summarize_stickslip(hist)
    @printf(
        "slider  H=%.0f m  k/k_crit=%.3f  (G/Wf)/k_crit=%.2f  τ_ss=%.3f MPa  ψ_ss=%.3f  Δψ=%.2f  V0=%.3e\n",
        p.H, p.k / p.k_crit, p.k_fault / p.k_crit, p.τ_ss / 1.0e6, p.ψ_ss, p.Δψ, p.V,
    )
    @printf(
        "slider  steps=%d  t=%.4f yr  events=%d  dt_load=%.3e s\n",
        length(hist), hist[end].t / YEAR, summary.n, p.dt_load,
    )
    for (i, (drop, vmax)) in enumerate(zip(summary.drops, summary.vmax))
        @printf("slider event %d  drop=%.3f MPa  Vpeak=%.3e m/s\n", i, drop / 1.0e6, vmax)
    end
    flush(stdout)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = seas_bp1_vw_params(; Δψ = 2.0)
    hist = integrate_slider(p; nsteps = 20_000, max_time = 200 * YEAR, dt_floor = 1.0, max_events = 2)
    report_slider(p, hist)
    if get(ENV, "SEAS_SLIDER_DYREL", "1") != "0"
        run_dyrel_stickslip(p; nx = 4, nsteps = 2_000, max_events = 1, adaptive = false, dt_floor = 1.0e-6)
    end
end
