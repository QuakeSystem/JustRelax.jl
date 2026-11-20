#=
Herrendörfer simple shear with the elastic eigenstrain rate-and-state split.

The fault is one vertex row. `Wf` is the LaMEM fault width (500 m here), the
thickness in `V / (2 Wf)`. SEAS BP1-FD also writes `Wf` for the down-dip height
of the rate-and-state section; that length is `Wf_height` and is not this width.
The host is Maxwell / elastic. After each DYREL solve the slip rate is the root
of the sinh law on `τ − (G Δt / Wf)(V − V_lag)`. That bracket, and the clearing
of the pseudo-velocity at the start of each solve, live in the shared kernels
compiled by the CUDA and AMDGPU modules. Inertia is on. Up to 8 passes are
allowed; the loop stops earlier once the rate changes by less than 1%.

The production mesh keeps a uniform 500 m spacing along the periodic x direction.
Around the fault, 4 km of 500 m cells (so the eigenstrain vertex sits in the
middle of that band). Toward the top and bottom the cells grow by a factor of
about 3. A jump much larger than that makes the DYREL step unstable once a
vertical velocity appears. The run stops after five rupture periods. In the
Herrendörfer reference that is about 100 years each, so about 500 years.

The fault starts locked, as in that run: state `ψ = φ = -1` and shear stress
zero, not the steady plate-rate stress. Stress then builds under the boundary
shear until the velocity-weakening patch fails. The physical step is the LaMEM
minimum of healing, weakening, and slip Courant time.

This is the second RSF path (`rsf_eigenstrain`). The viscosity-fold path in
`ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer.jl` keeps its own friction
update. Both drop the pseudo-velocity at the start of a rate-and-state solve.
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Printf

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

include(joinpath(@__DIR__, "simple_shear_setup.jl"))

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

const YEAR = 365.25 * 86400.0

"""Cell sizes that fill `L`, starting at `ratio0 * h_fine` and then growing at a nearly constant ratio."""
function geometric_outward(L, h_fine; n, ratio0 = 3.0)
    h1 = ratio0 * h_fine
    lo, hi = 1.01, 8.0
    for _ in 1:80
        mid = (lo + hi) / 2
        sum_mid = h1 * (mid^n - 1) / (mid - 1)
        sum_mid > L ? (hi = mid) : (lo = mid)
    end
    r = (lo + hi) / 2
    hs = [h1 * r^(k - 1) for k in 1:n]
    hs[end] += L - sum(hs)
    return hs, r
end

"""
Uniform `dx` along the periodic direction. In `y`, a `fine_height` band of `h_fine`
cells sits on the fault, and the mesh coarsens toward the top and bottom.
"""
function herrendorfer_graded_mesh(;
        x0 = -75.0e3, x1 = 75.0e3, dx = 500.0,
        y0 = -150.0e3, y1 = 0.0, y_fault = -75.0e3,
        h_fine = 500.0, fine_height = 4.0e3, n_outer = 4, ratio0 = 3.0,
    )
    nx = round(Int, (x1 - x0) / dx)
    abs((x1 - x0) - nx * dx) < 1.0e-6 || error("x-domain is not an integer number of dx=$dx cells")
    xv = collect(range(x0, x1; length = nx + 1))

    fine_height % h_fine == 0 || error("fine band $fine_height m is not a multiple of $h_fine m")
    n_fine = round(Int, fine_height / h_fine)
    iseven(n_fine) || error("fine band needs an even cell count so the fault sits on a vertex")
    y_fine0 = y_fault - fine_height / 2
    y_fine1 = y_fault + fine_height / 2
    (y0 < y_fine0 < y_fault < y_fine1 < y1) || error("fine band is outside the domain")

    h_bot, r_bot = geometric_outward(y_fine0 - y0, h_fine; n = n_outer, ratio0 = ratio0)
    h_top, r_top = geometric_outward(y1 - y_fine1, h_fine; n = n_outer, ratio0 = ratio0)
    yv = Float64[y0]
    y = y0
    for h in reverse(h_bot)
        y += h
        push!(yv, y)
    end
    for _ in 1:n_fine
        y += h_fine
        push!(yv, y)
    end
    for h in h_top
        y += h
        push!(yv, y)
    end
    yv[end] = y1
    dys = diff(yv)
    all(dys .> 0) || error("y vertices are not strictly increasing")
    ratios = dys[2:end] ./ dys[1:(end - 1)]
    jump = maximum(max.(ratios, 1 ./ ratios))
    jump < 3.6 || error("adjacent y cell-size ratio $jump is above 3.6")
    return (;
        xvi = (xv, yv), ni = (nx, length(yv) - 1),
        dy = dys, ratios = ratios, growth = (r_bot, r_top),
    )
end

"""Square cells of size `h` over the Herrendörfer domain. `y = -75 km` lands on a vertex for 250 m and 500 m."""
function herrendorfer_uniform_mesh(;
        x0 = -75.0e3, x1 = 75.0e3, y0 = -150.0e3, y1 = 0.0, h = 500.0,
    )
    nx = round(Int, (x1 - x0) / h)
    ny = round(Int, (y1 - y0) / h)
    abs((x1 - x0) - nx * h) < 1.0e-6 || error("x-domain is not an integer number of h=$h cells")
    abs((y1 - y0) - ny * h) < 1.0e-6 || error("y-domain is not an integer number of h=$h cells")
    return (;
        xvi = (collect(range(x0, x1; length = nx + 1)), collect(range(y0, y1; length = ny + 1))),
        ni = (nx, ny),
        h = h,
    )
end

function prepare_eigen_vtk!(dir, pvd_name)
    mkpath(dir)
    for f in readdir(dir)
        if startswith(f, "step_") || endswith(f, ".pvd") || endswith(f, ".vtu") ||
                endswith(f, ".vtr") || endswith(f, ".csv")
            rm(joinpath(dir, f); force = true)
        end
    end
    return joinpath(dir, pvd_name)
end

function write_eigen_vtk!(vtk_dir, pvd, it, t_s, xvi, xci, stokes, ez, Vx_v, Vy_v)
    velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
    tensor_invariant!(stokes.ε)
    # Velocity is the VTK vector. Vx and Vy are not written again as scalars.
    data_v = (;
        τxy = Array(stokes.τ.xy),
        εxy = Array(stokes.ε.xy),
        V_slip = Array(ez.V),
        ψ = Array(ez.ψ),
        a = Array(ez.a),
        b = Array(ez.b),
        fault = Array(ez.mask),
    )
    data_c = (;
        P = Array(stokes.P),
        εII = Array(stokes.ε.II),
    )
    path = joinpath(vtk_dir, "step_" * lpad(string(it), 6, '0'))
    save_vtk(path, xvi, xci, data_v, data_c, (Array(Vx_v), Array(Vy_v)); t = t_s, pvd = pvd)
    return nothing
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

function main(
        igg;
        xvi = nothing,
        nel_x = 16,
        coord_x = (-75.0e3, 75.0e3),
        Wf = 500.0,
        y_fault = -75.0e3,
        nel_y = (48, 1, 48),
        coord_y = (-150.0e3, y_fault - Wf / 2, y_fault + Wf / 2, 0.0),
        V_top = 4.0e-9,
        η0 = 5.0e26,
        G0 = 3.0e10,
        ν = 0.25,
        ρ = 2700.0,
        σ = 5.0e6,
        a = 0.011,
        b = 0.017,
        b_x = (-47000.0, -43000.0, 33000.0, 37000.0),
        b_val = (0.001, 0.017, 0.017, 0.001),
        μ₀ = 0.2,
        V₀ = 4.0e-9,
        L = 0.01,
        ψ_init = -1.0,
        τ_init = 0.0,
        x_probe = -50.0e3,
        nsteps = 3,
        n_ruptures = 0,
        max_years = 600.0,
        vtk_dir = nothing,
        pvd_name = "herrendorfer_eigenstrain",
    )
    if isnothing(xvi)
        mesh = segmented_grid_2D(;
            nel_x = nel_x, coord_x = coord_x, nel_y = nel_y, coord_y = coord_y,
        )
        xvi = mesh.xvi
        ni = mesh.ni
        dy_fault = abs(coord_y[3] - coord_y[2])
        dy_fault ≈ Wf ||
            error("fault segment thickness $dy_fault m is not Wf = $Wf m")
    else
        ni = (length(xvi[1]) - 1, length(xvi[2]) - 1)
    end
    nx, ny = ni
    grid = Geometry(PTArray(backend), xvi...)

    rheology = init_host_rheology(; η = η0, G = G0, ν = ν, ρ = ρ)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    proto = RateStateFriction(0.0, μ₀, V₀, a, b, L, 0.0, Wf)
    # Herrendörfer initial state: locked fault, φ = −1, shear stress still zero.
    # Steady sliding at the plate rate would already be the failed state.
    ez = eigenstrain_rsf(
        backend, ni; Wf = Wf, G = G0, σ = σ, law = proto, tol = 1.0e-2, max_pass = 8,
    )
    j_fault = init_eigenstrain_row!(
        ez, grid.xvi;
        y_fault = y_fault, a = a, b = b, ψ = ψ_init, V_lag = 0.0,
        b_x = b_x, b_val = b_val,
    )

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, τ_init)
    fill!(stokes.τ_o.xy, τ_init)

    ybot, ytop = xvi[2][1], xvi[2][end]
    yVx = Array(grid.xi_vel[1][2])
    stokes.V.Vx .= PTArray(backend)(
        [
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi[1], y in yVx
        ]
    )
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    fill!(stokes.V0.Vx, 0.0)
    fill!(stokes.V0.Vy, 0.0)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dt_min = 1.0e-2
    dt_max = 1.0e9
    dt = clamp(eigenstrain_lamem_dt(ez; ν = ν), dt_min, dt_max)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6, CFL = 0.9)

    yv = Array(grid.xvi[2])
    dy_lo = j_fault > 1 ? yv[j_fault] - yv[j_fault - 1] : NaN
    dy_hi = j_fault < length(yv) ? yv[j_fault + 1] - yv[j_fault] : NaN
    i_probe = argmin(abs.(Array(grid.xvi[1]) .- x_probe))
    println(
        "eigenstrain fault row j=$j_fault  y=$(yv[j_fault]) m  " *
            "local dy=($dy_lo, $dy_hi) m  Wf=$Wf m  ni=$ni  " *
            "ψ_init=$ψ_init  τ_init=$τ_init Pa  dt=$dt s",
    )
    println(
        "probe x=$(Array(grid.xvi[1])[i_probe]) m (Herrendörfer x = 25 km)  " *
            "y=$(yv[j_fault]) m  b=$(Array(ez.b)[i_probe, j_fault])",
    )
    dy_lo ≈ dy_hi ||
        error("fault vertex cells differ (dy = $dy_lo, $dy_hi)")

    xvi_h = Array.(grid.xvi)
    xci_h = Array.(grid.xci)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    pvd = nothing
    hist = nothing
    probe = nothing
    if !isnothing(vtk_dir)
        pvd = prepare_eigen_vtk!(vtk_dir, pvd_name)
        hist = open(joinpath(vtk_dir, "history.csv"), "w")
        println(hist, "step,t_year,t_s,events,in_event,passes,rel,Vmax,dt,tau_min,tau_max,wall_s")
        flush(hist)
        probe = open(joinpath(vtk_dir, "point_xm50km_z75000m.csv"), "w")
        println(probe, "Time,Dt,Sliprate,Phi,SII")
        println(probe, "0.0,0.0,0.0,$ψ_init,$τ_init")
        flush(probe)
    end

    # A slip-rate spike above this is coseismic; falling back below the quiet
    # threshold closes the event. Five closed events is the stopping point.
    V_event = 1.0e-3
    V_quiet = 1.0e-5
    t = 0.0
    n_events = 0
    in_event = false
    it = 0
    try
        if !isnothing(pvd)
            write_eigen_vtk!(vtk_dir, pvd, 0, 0.0, xvi_h, xci_h, stokes, ez, Vx_v, Vy_v)
            println("ParaView: $(pvd).pvd   (time axis is seconds)")
            flush(stdout)
        end
        while it < nsteps
            it += 1
            wall0 = time()
            out = eigenstrain_physical_step!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg, ez;
                inertia = true,
                ρ_inertia = ρ,
                solve_kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 5.0e2,
                    iterMax_PH = 6,
                    nout = 50,
                    rel_drop = 1.0e-3,
                    linear_viscosity = true,
                    update_material = false,
                ),
            )
            @views stokes.V.Vx[:, end] .= V_top
            t += dt
            wall = time() - wall0
            τ_row = Array(stokes.τ.xy)[:, j_fault]
            isfinite(out.V_max) && isfinite(out.rel) && all(isfinite, τ_row) ||
                error("non-finite state at it=$it t=$t s")
            τmin, τmax = extrema(τ_row)
            V_probe = Array(ez.V)[i_probe, j_fault]
            ϕ_probe = Array(ez.ψ)[i_probe, j_fault]
            τ_probe = τ_row[i_probe]
            t_year = t / YEAR
            if !in_event && out.V_max > V_event
                in_event = true
                println("rupture $(n_events + 1) starts at t=$(round(t_year; digits = 4)) yr  Vmax=$(out.V_max)")
            elseif in_event && out.V_max < V_quiet
                in_event = false
                n_events += 1
                println("rupture $n_events ends at t=$(round(t_year; digits = 4)) yr  Vmax=$(out.V_max)")
            end
            @printf(
                "it=%d  t=%9.4f yr  events=%d  passes=%d  rel=%.3e  Vmax=%.3e  dt=%.3e s  wall=%.2f s  τ=(%.6e, %.6e)  probe V=%.3e φ=%.4f τ=%.6e\n",
                it, t_year, n_events, out.passes, out.rel, out.V_max, dt, wall, τmin, τmax,
                V_probe, ϕ_probe, τ_probe,
            )
            flush(stdout)
            if !isnothing(hist)
                println(
                    hist,
                    "$it,$t_year,$t,$n_events,$(Int(in_event)),$(out.passes),$(out.rel),$(out.V_max),$dt,$τmin,$τmax,$wall",
                )
                flush(hist)
            end
            if !isnothing(probe)
                println(probe, "$t,$dt,$V_probe,$ϕ_probe,$τ_probe")
                flush(probe)
            end
            # One frame every 10 steps, in the interseismic build-up and during slip.
            if !isnothing(pvd) && rem(it, 10) == 0
                write_eigen_vtk!(
                    vtk_dir, pvd, it, t, xvi_h, xci_h, stokes, ez, Vx_v, Vy_v,
                )
            end
            if n_ruptures > 0 && n_events >= n_ruptures
                println("stopped after $n_events rupture periods at t=$(round(t_year; digits = 4)) yr")
                break
            end
            if n_ruptures > 0 && t_year >= max_years
                println("stopped at the $max_years yr cap with $n_events rupture periods")
                break
            end
            dt = clamp(eigenstrain_lamem_dt(ez; ν = ν), dt_min, dt_max)
            args = (; args..., dt = dt)
        end
        if !isnothing(pvd)
            write_eigen_vtk!(
                vtk_dir, pvd, it, t, xvi_h, xci_h, stokes, ez, Vx_v, Vy_v,
            )
            println("ParaView collection: $(pvd).pvd")
            println("history: $(joinpath(vtk_dir, "history.csv"))")
            println("probe:   $(joinpath(vtk_dir, "point_xm50km_z75000m.csv"))")
            flush(stdout)
        end
    finally
        isnothing(hist) || close(hist)
        isnothing(probe) || close(probe)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    Wf = 500.0
    y_fault = -75.0e3
    mesh = herrendorfer_graded_mesh(;
        x0 = -75.0e3, x1 = 75.0e3, dx = 500.0,
        y0 = -150.0e3, y1 = 0.0, y_fault = y_fault,
        h_fine = Wf, fine_height = 4.0e3, n_outer = 4, ratio0 = 3.0,
    )
    dys = mesh.dy
    ratios = mesh.ratios
    println("ni = $(mesh.ni)   dx = 500 m (uniform, periodic)")
    println("dy [m], bottom → top:")
    println(join(round.(dys; digits = 2), ", "))
    println("adjacent dy ratio (upper/lower), growth factors $(mesh.growth):")
    println(join(round.(ratios; digits = 3), ", "))
    flush(stdout)
    nx, ny = mesh.ni
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    vtk_dir = joinpath(@__DIR__, "VTK_eigenstrain_herrendorfer", "vtk")
    main(
        igg;
        xvi = mesh.xvi,
        Wf = Wf,
        y_fault = y_fault,
        nsteps = 1_000_000,
        n_ruptures = 5,
        max_years = 600.0,
        vtk_dir = vtk_dir,
        pvd_name = "herrendorfer_eigenstrain",
    )
end
