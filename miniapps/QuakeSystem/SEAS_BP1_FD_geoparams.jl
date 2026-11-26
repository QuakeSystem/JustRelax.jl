#=
SEAS BP1-FD (SCEC): 2D antiplane strike-slip, full inertia, GeoParams rate-and-state.

JustRelax has no out-of-plane unknown, so the benchmark is the 3D solver with
slip Vx(y, z). ImplicitGlobalGrid rejects a grid with `nx = 1`, so the strike
direction is two equal periodic cells. The fields do not vary in x. That is the
same reduction LaMEM uses, and it is the native 3D momentum residual

    ∂τxy/∂y + ∂τxz/∂z = ρ ∂Vx/∂t

with Vy = Vz = 0. Free-slip on the top and bottom faces is ∂Vx/∂z = 0, which is
the antiplane free surface (and the artificial base).

The fault is the xy edges at y = 0. GeoParams is a scalar law; the 3D stress
kernel would otherwise ignore it, because it builds τxy from a harmonic mean of
cell viscosities. `RateStateXY` writes the series viscosity onto those edges
and the kernel uses it there. Cell viscosities stay elastic, so the depth shear
τxz does not become frictional.

Benchmark (not a literal copy of the LaMEM VelBoxes):
  RSF from the free surface down to 40 km, a(z) = 0.010 above 15 km, linear to
  0.025 at 18 km, then 0.025. Below 40 km the two Vx nodes that form the fault
  jump are fixed at ±Vp/2. The outer fault-normal rows are fixed at the same
  plate rate. Mesh segments and the 100 m fault spacing are the LaMEM 100 m grid.

DYREL controls match the Herrendörfer GeoParams miniapp: iterMax = 5000,
rel_drop = 1e-4, ϵ = 1e-8, linear viscosity, inertia on.

GPU: set `isCUDA = true` below (the ParallelStencil backend is fixed at compile
time) and launch with 8 threads.

    julia -t 8 --project=. miniapps/QuakeSystem/SEAS_BP1_FD_geoparams.jl --smoke
    julia -t 8 --project=. miniapps/QuakeSystem/SEAS_BP1_FD_geoparams.jl --run

`--run` writes one time series per step for the fault at 10 km and 12.5 km:
`VTK/seas_bp1_fd_fault_10km.csv` and `VTK/seas_bp1_fd_fault_12.5km.csv`.
The other benchmark depths stay in `VTK/seas_bp1_fd_stations.csv`.
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D
using Printf

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

using ParallelStencil

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

using GeoParams

const CS = 3464.0
const RHO = 2670.0
const G0 = RHO * CS^2
const YEAR = 365.25 * 86400.0
const STATIONS_KM = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 35.0)
const FAULT_STATION_KM = (10.0, 12.5)

"""LaMEM `MeshSeg1D`: vertex coordinates, optional per-segment bias."""
function mesh_seg_1d(nel, coord; bias = nothing)
    nel_v = nel isa Integer ? (Int(nel),) : Tuple(Int(n) for n in nel)
    coord_v = Tuple(Float64(c) for c in coord)
    nseg = length(nel_v)
    length(coord_v) == nseg + 1 ||
        throw(ArgumentError("coord must have length $(nseg + 1), got $(length(coord_v))"))
    bias_v = if isnothing(bias)
        ntuple(_ -> 1.0, nseg)
    else
        length(bias) == nseg ||
            throw(ArgumentError("bias must have length $nseg, got $(length(bias))"))
        Tuple(Float64(b) for b in bias)
    end
    xvi = Vector{Float64}(undef, sum(nel_v) + 1)
    xvi[1] = coord_v[1]
    inode = 1
    @inbounds for iseg in 1:nseg
        M = nel_v[iseg]
        xstart, xclose = coord_v[iseg], coord_v[iseg + 1]
        b = bias_v[iseg]
        avg = (xclose - xstart) / M
        if b == 1.0
            for k in 1:M
                inode += 1
                xvi[inode] = xstart + k * avg
            end
        else
            begSz = 2 * avg / (1 + b)
            endSz = b * begSz
            dx = M > 1 ? (endSz - begSz) / (M - 1) : 0.0
            x = xstart
            for k in 0:(M - 1)
                x += begSz + k * dx
                inode += 1
                xvi[inode] = x
            end
        end
        xvi[inode] = xclose
    end
    return xvi
end

function bp1_rheology(; η = 5.0e26, D = 100.0)
    el = ConstantElasticity(; G = G0, ν = 0.25)
    visc = LinearViscous(; η = η)
    # a on the law is the surface value; a(z) is stored on the fault edges.
    rsf = GeoParams.RateStateFriction(0.0, 0.6, 1.0e-6, 0.010, 0.015, 0.008, 0.0, D)
    return (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = RHO),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, rsf, el)),
            Elasticity = el,
        ),
    )
end

@parallel_indices (i, j, k) function _fill_phase1_3D!(phases)
    @index phases[1, i, j, k] = 1.0
    return nothing
end

function fill_phase1!(phase_ratios)
    for loc in (phase_ratios.center, phase_ratios.vertex, phase_ratios.xy, phase_ratios.xz, phase_ratios.yz, phase_ratios.Vx, phase_ratios.Vy, phase_ratios.Vz)
        @parallel (@idx size(loc)) _fill_phase1_3D!(loc)
    end
    return nothing
end

function station_rows(zc)
    rows = NamedTuple[]
    for km in STATIONS_KM
        k = argmin(abs.(zc .+ km * 1.0e3))
        push!(rows, (; km, k, depth_m = -zc[k]))
    end
    return rows
end

function open_station_csv(path)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, "# problem=SEAS BP1-FD")
    println(io, "# code=JustRelax GeoParams rate-and-state, 3D antiplane, two periodic strike cells")
    println(io, "# slip_rate and state are log10. shear_stress is MPa. depth_m is positive downward.")
    println(io, "t,depth_m,slip,slip_rate,shear_stress,state,Vp")
    flush(io)
    return io
end

function log10_theta(Ω, L, V0)
    return log10(L / V0) + Ω / log(10)
end

function fault_station_path(km)
    label = km == floor(km) ? string(Int(km)) : string(km)
    return joinpath(@__DIR__, "VTK", "seas_bp1_fd_fault_$(label)km.csv")
end

function open_fault_station_csv(path, km, depth_m, z_m)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, "# problem=SEAS BP1-FD")
    println(io, "# station=fault y=0  target_depth=$(km) km  cell_center_depth=$(depth_m) m  z=$(z_m) m")
    println(io, "# slip_rate and state are log10. shear_stress is MPa. depth_m is positive downward.")
    println(io, "t,depth_m,y_m,z_m,slip,slip_rate,shear_stress,state,Vp")
    flush(io)
    return io
end

function fault_sample(stokes, fields, j, k, L, V0)
    Vj = Float64(stokes.V.Vx[2, j + 1, k + 1] - stokes.V.Vx[2, j, k + 1])
    return (;
        Vj = Vj,
        τ = Float64(stokes.τ.xy[2, j, k]),
        Ω = Float64(fields.Ω[2, j, k]),
        Vp = Float64(fields.Vp[2, j, k]),
        rate = abs(Vj) > 0 ? log10(abs(Vj)) : -Inf,
        state = log10_theta(Float64(fields.Ω[2, j, k]), L, V0),
    )
end

function write_fault_row!(io, t, station, slip, sample)
    println(
        io,
        "$t,$(station.depth_m),0.0,$(-station.depth_m),$slip,$(sample.rate),$(sample.τ / 1.0e6),$(sample.state),$(sample.Vp)",
    )
    return nothing
end

"""
    Build the BP1-FD grid, rheology, fault state, plate velocity, and prestress.
Does not enter the time loop. Halo exchange needs the global grid, so this
creates it when it is not already active.
"""
function prepare_bp1(;
        η = 5.0e26,
        D = 100.0,
        σ = 50.0e6,
        Vp = 1.0e-9,
        dt = 2522.0,
    )
    igg = if JustRelax.ImplicitGlobalGrid.grid_is_initialized()
        gg = get_global_grid()
        IGG(gg.me, gg.dims, gg.nprocs, gg.coords, gg.comm)
    else
        IGG(init_global_grid(2, 210, 224; init_MPI = !JustRelax.MPI.Initialized())...)
    end
    # ImplicitGlobalGrid rejects nx = 1. Two equal cells are the smallest strike grid.
    xv = mesh_seg_1d(2, (-50.0, 50.0))
    yv = mesh_seg_1d((100, 10, 100), (-1.0e6, -500.0, 500.0, 1.0e6); bias = (0.00502765, 1.0, 198.9))
    zv = mesh_seg_1d((24, 200), (-73000.0, -20000.0, 0.0); bias = (0.02272727, 1.0))
    ni = (length(xv) - 1, length(yv) - 1, length(zv) - 1)
    grid = Geometry(PTArray(backend), xv, yv, zv)
    rheology = bp1_rheology(; η = η, D = D)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    fill_phase1!(phase_ratios)

    fields = RateStateXY(ni; skip_vs = true)
    meta = init_antiplane_fault!(
        fields, grid.xvi, grid.xci;
        z_rsf_min = -40.0e3,
        a_knots = (-18.0e3, -15.0e3),
        a_vals = (0.025, 0.010),
        b = 0.015, μ₀ = 0.6, V₀ = 1.0e-6, V_plate = Vp, σ = σ,
    )
    flow_bcs = VelocityBoundaryConditions(;
        no_slip = (left = false, right = false, front = false, back = false, bot = false, top = false),
        free_slip = (left = false, right = false, front = false, back = false, bot = true, top = true),
        periodic = (left = true, right = true, front = false, back = false, bot = false, top = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, meta.τ0)
    fill!(stokes.τ_o.xy, meta.τ0)

    Vx = zeros(size(stokes.V.Vx))
    @inbounds for j in 1:size(Vx, 2)
        Vx[:, j, :] .= j <= meta.j_fault ? -meta.half : meta.half
    end
    copyto!(stokes.V.Vx, Vx)
    fill!(stokes.V.Vy, 0.0)
    fill!(stokes.V.Vz, 0.0)
    copyto!(stokes.V0.Vx, stokes.V.Vx)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    viscosity_cutoff = (1.0, η)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    return (; igg, grid, ni, rheology, phase_ratios, fields, meta, flow_bcs, stokes, args, viscosity_cutoff, D, Vp, σ, dt)
end

function smoke_bp1()
    case = prepare_bp1()
    (; ni, grid, fields, meta, stokes, phase_ratios, rheology, args, viscosity_cutoff) = case
    yv = Array(grid.xvi[2])
    zc = Array(grid.xci[3])
    j = meta.j_fault
    dy_left = yv[j] - yv[j - 1]
    dy_right = yv[j + 1] - yv[j]
    ni == (2, 210, 224) || error("expected cell counts (2, 210, 224), got $ni")
    isapprox(dy_left, 100.0; atol = 1.0e-6) || error("fault cell left of y=0 is $dy_left m, not 100 m")
    isapprox(dy_right, 100.0; atol = 1.0e-6) || error("fault cell right of y=0 is $dy_right m, not 100 m")
    a = Array(fields.a_eff)
    mask = Array(fields.rsf_mask)
    k_surf = argmin(abs.(zc))
    k_20 = argmin(abs.(zc .+ 20.0e3))
    k_deep = argmin(abs.(zc .+ 50.0e3))
    isapprox(a[1, j, k_surf], 0.010; atol = 1.0e-8) || error("surface a is $(a[1, j, k_surf])")
    isapprox(a[1, j, k_20], 0.025; atol = 1.0e-8) || error("a at ~20 km is $(a[1, j, k_20])")
    mask[1, j, k_deep] == 0.0 || error("rate-and-state is still on below 40 km")
    mask[1, j, k_surf] == 1.0 || error("rate-and-state is off at the surface")
    any(>(0.5), mask[:, 1, :]) && error("rate-and-state leaked off the fault")

    apply_geoparams_rsf_viscosity!(
        stokes, fields, phase_ratios, rheology, case.dt; viscosity_cutoff = viscosity_cutoff,
    )
    ηe = Array(fields.η)
    all(isfinite, ηe[mask .> 0.5]) && all(>(0), ηe[mask .> 0.5]) ||
        error("fault-edge viscosity is not a positive finite number")
    maximum(ηe[mask .≤ 0.5]) == 0.0 || error("off-fault xy viscosity was overwritten")

    fixed = Array(fields.vx_fixed)
    value = Array(fields.vx_value)
    # Outer fault-normal row and a deep fault node stay at the plate rate.
    isapprox(value[1, 2, 2], -0.5e-9; rtol = 1.0e-12) || error("negative flank is not -Vp/2")
    isapprox(value[1, end - 1, 2], 0.5e-9; rtol = 1.0e-12) || error("positive flank is not +Vp/2")
    fixed[1, j, k_deep + 1] > 0.5 || error("deep fault Vx is not fixed")
    println(
        "BP1-FD smoke OK  ni=$ni  Wf_cells=100 m  τ0=$(meta.τ0 / 1.0e6) MPa  " *
            "RSF edges=$(meta.n_rsf)  G=$G0  threads=$(Threads.nthreads())",
    )
    println("stations (cell-center depth, m): ", [s.depth_m for s in station_rows(zc)])
    return case
end

function run_bp1(;
        nsteps = 245000,
        t_max = 1500.0 * YEAR,
        dt = 2522.0,
        dt_min = 5.0e-3,
        dt_max = 1.0e9,
        iterMax = 50.0e2,
        rel_drop = 1.0e-4,
        ϵ = 1.0e-8,
        print_every = 1,
        csv_path = joinpath(@__DIR__, "VTK", "seas_bp1_fd_stations.csv"),
    )
    case = smoke_bp1()
    (; igg, ni, grid, rheology, phase_ratios, fields, meta, flow_bcs, stokes, args, viscosity_cutoff) = case
    ni == (2, 210, 224) || error("run grid $ni does not match the MPI grid")
    zc = Array(grid.xci[3])
    stations = station_rows(zc)
    fault_stations = [s for s in stations if s.km in FAULT_STATION_KM]
    length(fault_stations) == length(FAULT_STATION_KM) ||
        error("fault stations $(FAULT_STATION_KM) were not found on the grid")
    slip = zeros(length(stations))
    fault_slip = zeros(length(fault_stations))
    csv = open_station_csv(csv_path)
    fault_csv = [
        open_fault_station_csv(fault_station_path(s.km), s.km, s.depth_m, -s.depth_m)
        for s in fault_stations
    ]
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = ϵ)
    ρ_inertia = @zeros(ni...)
    compute_ρ!(ρ_inertia, phase_ratios, rheology, args)
    j = meta.j_fault
    L = 0.008
    V0 = 1.0e-6
    t = 0.0
    it = 0
    out = nothing
    try
        for (s, station) in enumerate(fault_stations)
            sample = fault_sample(stokes, fields, j, station.k, L, V0)
            write_fault_row!(fault_csv[s], 0.0, station, 0.0, sample)
            flush(fault_csv[s])
        end
        while it < nsteps && t < t_max
            out = solve_DYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
                kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = iterMax,
                    nout = 50,
                    rel_drop = rel_drop,
                    linear_viscosity = true,
                    viscosity_cutoff = viscosity_cutoff,
                    gp_rsf = fields,
                    inertia = true,
                    ρ_inertia = ρ_inertia,
                ),
            )
            it += 1
            t += dt
            for (s, station) in enumerate(stations)
                sample = fault_sample(stokes, fields, j, station.k, L, V0)
                slip[s] += sample.Vj * dt
                println(
                    csv,
                    "$t,$(station.depth_m),$(slip[s]),$(sample.rate),$(sample.τ / 1.0e6),$(sample.state),$(sample.Vp)",
                )
            end
            flush(csv)
            fault_line = String[]
            for (s, station) in enumerate(fault_stations)
                sample = fault_sample(stokes, fields, j, station.k, L, V0)
                fault_slip[s] += sample.Vj * dt
                write_fault_row!(fault_csv[s], t, station, fault_slip[s], sample)
                flush(fault_csv[s])
                push!(fault_line, "$(station.km)km τ=$(sample.τ / 1.0e6) MPa V=$(sample.rate)")
            end
            if it == 1 || rem(it, print_every) == 0
                println(
                    "it=$it  t=$t  Vp_max=$(out.Vp_max)  dt=$dt  dt_rsf=$(out.dt_rsf)  ",
                    join(fault_line, "  "),
                )
            end
            if isfinite(out.dt_rsf) && out.dt_rsf < 1.0e9
                dt = clamp(out.dt_rsf, dt_min, dt_max)
                args = (; args..., dt = dt)
            end
        end
    finally
        close(csv)
        foreach(close, fault_csv)
        JustRelax.MPI.Initialized() && finalize_global_grid(; finalize_MPI = false)
    end
    println("finished it=$it t=$t")
    println("  all depths → $csv_path")
    for station in fault_stations
        println("  fault $(station.km) km ($(station.depth_m) m) → $(fault_station_path(station.km))")
    end
    return (; case, out, t, it, Vp)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--run" in ARGS
        run_bp1()
    else
        smoke_bp1()
        println("connectivity only. Start the time loop with --run")
    end
end
