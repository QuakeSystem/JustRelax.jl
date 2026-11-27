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
  plate rate. The domain is 160 km from the fault to each side and 160 km deep.
  Cells are 25 m inside ±1 km of the fault and from the surface to 20 km depth.
  Outside that, 200 cells on each side and 200 cells below 20 km grow smoothly.
  Wf is 25 m, the same as the fine cell size.

DYREL controls match the Herrendörfer GeoParams miniapp: iterMax = 5000,
rel_drop = 1e-4, ϵ = 1e-8, linear viscosity, inertia on.

GPU: set `isCUDA = true` below (the ParallelStencil backend is fixed at compile
time) and launch with 8 threads.

    julia -t 8 --project=. miniapps/QuakeSystem/SEAS_BP1_FD_geoparams.jl --smoke
    julia -t 8 --project=. miniapps/QuakeSystem/SEAS_BP1_FD_geoparams.jl --run

`--run` integrates to 3e10 s. Output is SI: time in s, slip in m, slip rate and
`Vp` in m/s, stress in Pa, and the state variable θ in s. Each CSV starts with
a header whose first column is `Time`, so ParaView can use it as a time series.
Snapshots go to `VTK/seas_bp1_fd.pvd` every 10 steps, with the fields LaMEM
writes for this benchmark: phase, velocity, j2_dev_stress, j2_strain_rate,
state_rsf, state_rsf_cell, Vp_rsf, Vp_rsf_cell.
=#
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D
using WriteVTK
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

"""
    bp1_mesh()

Fault-normal coordinate from -160 km to 160 km: 200 growing cells, then 2 km of
25 m cells centered on the fault, then 200 growing cells. Depth from -160 km to
the surface: 200 growing cells up to -20 km, then 25 m cells to z = 0.
"""
function bp1_mesh()
    dy = 25.0
    y_fine = 1.0e3
    y_far = 160.0e3
    n_side = 200
    n_fine = round(Int, 2 * y_fine / dy)
    dy_far = 2 * (y_far - y_fine) / n_side - dy
    z_fine = -20.0e3
    z_bot = -160.0e3
    n_deep = 200
    n_shallow = round(Int, -z_fine / dy)
    dz_bot = 2 * (z_fine - z_bot) / n_deep - dy
    xv = mesh_seg_1d(2, (-dy, dy))
    yv = mesh_seg_1d(
        (n_side, n_fine, n_side),
        (-y_far, -y_fine, y_fine, y_far);
        bias = (dy / dy_far, 1.0, dy_far / dy),
    )
    zv = mesh_seg_1d(
        (n_deep, n_shallow),
        (z_bot, z_fine, 0.0);
        bias = (dy / dz_bot, 1.0),
    )
    return xv, yv, zv
end

function bp1_rheology(; η = 5.0e26, D = 25.0)
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
    println(io, "Time,depth_m,slip,slip_rate,shear_stress,state,Vp")
    flush(io)
    return io
end

function theta_seconds(Ω, L, V0)
    return (L / V0) * exp(Ω)
end

function fault_station_path(km)
    label = km == floor(km) ? string(Int(km)) : string(km)
    return joinpath(@__DIR__, "VTK", "seas_bp1_fd_fault_$(label)km.csv")
end

function open_fault_station_csv(path)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, "Time,depth_m,y_m,z_m,slip,slip_rate,shear_stress,state,Vp")
    flush(io)
    return io
end

function host_float(A, i, j, k)
    A isa Array && return Float64(@inbounds A[i, j, k])
    return Float64(first(Array(@view A[i:i, j:j, k:k])))
end

function fault_sample(stokes, fields, j, k, L, V0)
    Vj = host_float(stokes.V.Vx, 2, j + 1, k + 1) - host_float(stokes.V.Vx, 2, j, k + 1)
    Ω = host_float(fields.Ω, 2, j, k)
    return (;
        Vj = Vj,
        τ = host_float(stokes.τ.xy, 2, j, k),
        Ω = Ω,
        Vp = host_float(fields.Vp, 2, j, k),
        rate = Vj,
        state = theta_seconds(Ω, L, V0),
    )
end

function write_fault_row!(io, t, station, slip, sample)
    println(
        io,
        "$t,$(station.depth_m),0.0,$(-station.depth_m),$slip,$(sample.rate),$(sample.τ),$(sample.state),$(sample.Vp)",
    )
    return nothing
end

function vertex_velocity(stokes, ni)
    Vx = Array(stokes.V.Vx)
    Vy = Array(stokes.V.Vy)
    Vz = Array(stokes.V.Vz)
    nx, ny, nz = ni
    n = (nx + 1, ny + 1, nz + 1)
    Vxv = Array{Float64}(undef, n)
    Vyv = similar(Vxv)
    Vzv = similar(Vxv)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        Vxv[i, j, k] = 0.25 * (Vx[i, j, k] + Vx[i, j + 1, k] + Vx[i, j, k + 1] + Vx[i, j + 1, k + 1])
        Vyv[i, j, k] = 0.25 * (Vy[i, j, k] + Vy[i + 1, j, k] + Vy[i, j, k + 1] + Vy[i + 1, j, k + 1])
        Vzv[i, j, k] = 0.25 * (Vz[i, j, k] + Vz[i, j + 1, k] + Vz[i + 1, j, k] + Vz[i + 1, j + 1, k])
    end
    return Vxv, Vyv, Vzv
end

function xy_edge_state(edge, mask, L, V0)
    state = zeros(size(edge))
    @inbounds for i in eachindex(edge, mask, state)
        mask[i] > 0.5 || continue
        state[i] = theta_seconds(edge[i], L, V0)
    end
    return state
end

function xy_edge_to_vertex(edge, ni)
    nx, ny, nz = ni
    v = zeros(nx + 1, ny + 1, nz + 1)
    @inbounds for k in 1:nz, j in 1:(ny + 1), i in 1:(nx + 1)
        v[i, j, k] = max(v[i, j, k], edge[i, j, k])
        v[i, j, k + 1] = max(v[i, j, k + 1], edge[i, j, k])
    end
    return v
end

function xy_edge_to_cell(edge, ni)
    nx, ny, nz = ni
    c = zeros(nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c[i, j, k] = max(edge[i, j, k], edge[i + 1, j, k], edge[i, j + 1, k], edge[i + 1, j + 1, k])
    end
    return c
end

function reset_vtk(vtk_dir, pvd)
    mkpath(vtk_dir)
    rm(pvd * ".pvd"; force = true)
    for path in readdir(vtk_dir; join = true)
        name = basename(path)
        startswith(name, "seas_bp1_fd_") || continue
        endswith(name, ".vtr") || endswith(name, ".vts") || continue
        rm(path; force = true)
    end
    return nothing
end

function write_paraview(case, it, t, pvd, L, V0)
    (; grid, ni, stokes, fields) = case
    tensor_invariant!(stokes.τ)
    tensor_invariant!(stokes.ε)
    Vxv, Vyv, Vzv = vertex_velocity(stokes, ni)
    mask = Array(fields.rsf_mask)
    state_e = xy_edge_state(Array(fields.Ω), mask, L, V0)
    Vp_e = Array(fields.Vp) .* mask
    velocity = zeros(Float32, 3, (ni .+ 1)...)
    velocity[1, :, :, :] .= Float32.(Vxv)
    velocity[2, :, :, :] .= Float32.(Vyv)
    velocity[3, :, :, :] .= Float32.(Vzv)
    xvi = ntuple(d -> Array(grid.xvi[d]), 3)
    fname = joinpath(dirname(pvd), "seas_bp1_fd_" * lpad(string(it), 8, '0'))
    vtk_grid(fname, xvi...) do vtk
        vtk["phase", VTKCellData()] = fill(Float32(1), ni)
        vtk["j2_dev_stress", VTKCellData()] = Float32.(Array(stokes.τ.II))
        vtk["j2_strain_rate", VTKCellData()] = Float32.(Array(stokes.ε.II))
        vtk["state_rsf_cell", VTKCellData()] = Float32.(xy_edge_to_cell(state_e, ni))
        vtk["Vp_rsf_cell", VTKCellData()] = Float32.(xy_edge_to_cell(Vp_e, ni))
        vtk["state_rsf", VTKPointData()] = Float32.(xy_edge_to_vertex(state_e, ni))
        vtk["Vp_rsf", VTKPointData()] = Float32.(xy_edge_to_vertex(Vp_e, ni))
        vtk["velocity", VTKPointData()] = velocity
        vtk["TimeValue"] = t
        paraview_collection(pvd; append = true) do collection
            collection_add_timestep(collection, vtk, t)
        end
    end
    return fname
end

"""
    Build the BP1-FD grid, rheology, fault state, plate velocity, and prestress.
Does not enter the time loop. Halo exchange needs the global grid, so this
creates it when it is not already active.
"""
function prepare_bp1(;
        η = 5.0e26,
        D = 25.0,
        σ = 50.0e6,
        Vp = 1.0e-9,
        dt = 2522.0,
    )
    # ImplicitGlobalGrid rejects nx = 1. Two equal cells are the smallest strike grid.
    xv, yv, zv = bp1_mesh()
    ni = (length(xv) - 1, length(yv) - 1, length(zv) - 1)
    igg = if JustRelax.ImplicitGlobalGrid.grid_is_initialized()
        (nx_g(), ny_g(), nz_g()) == ni ||
            error("MPI grid $((nx_g(), ny_g(), nz_g())) does not match the mesh $ni")
        gg = get_global_grid()
        IGG(gg.me, gg.dims, gg.nprocs, gg.coords, gg.comm)
    else
        IGG(init_global_grid(ni...; init_MPI = !JustRelax.MPI.Initialized())...)
    end
    grid = Geometry(PTArray(backend), xv, yv, zv)
    rheology = bp1_rheology(; η = η, D = D)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    fill_phase1!(phase_ratios)

    fields = RateStateXY(backend, ni; skip_vs = true)
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
    ni == (2, 480, 1000) || error("expected cell counts (2, 480, 1000), got $ni")
    isapprox(dy_left, 25.0; atol = 1.0e-6) || error("fault cell left of y=0 is $dy_left m, not 25 m")
    isapprox(dy_right, 25.0; atol = 1.0e-6) || error("fault cell right of y=0 is $dy_right m, not 25 m")
    isapprox(yv[1], -160.0e3; atol = 1.0e-4) || error("fault-normal domain starts at $(yv[1]) m")
    isapprox(yv[end], 160.0e3; atol = 1.0e-4) || error("fault-normal domain ends at $(yv[end]) m")
    zv = Array(grid.xvi[3])
    isapprox(zv[1], -160.0e3; atol = 1.0e-4) || error("domain bottom is $(zv[1]) m")
    isapprox(zv[end] - zv[end - 1], 25.0; atol = 1.0e-6) ||
        error("surface cell is $(zv[end] - zv[end - 1]) m, not 25 m")
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
        "BP1-FD smoke OK  ni=$ni  Wf=$(case.D) m  cells=25 m  τ0=$(meta.τ0) Pa  " *
            "RSF edges=$(meta.n_rsf)  G=$G0  threads=$(Threads.nthreads())",
    )
    println("stations (cell-center depth, m): ", [s.depth_m for s in station_rows(zc)])
    return case
end

function run_bp1(;
        nsteps = 1_000_000,
        t_max = 3.0e10,
        dt = 2522.0,
        dt_min = 5.0e-3,
        dt_max = 1.0e9,
        nstep_out = 10,
        iterMax = 50.0e2,
        rel_drop = 1.0e-4,
        ϵ = 1.0e-8,
        print_every = 1,
        csv_path = joinpath(@__DIR__, "VTK", "seas_bp1_fd_stations.csv"),
        pvd = joinpath(@__DIR__, "VTK", "seas_bp1_fd"),
    )
    case = smoke_bp1()
    (; igg, ni, grid, rheology, phase_ratios, fields, meta, flow_bcs, stokes, args, viscosity_cutoff) = case
    ni == (2, 480, 1000) || error("run grid $ni does not match the 25 m mesh")
    zc = Array(grid.xci[3])
    stations = station_rows(zc)
    fault_stations = [s for s in stations if s.km in FAULT_STATION_KM]
    length(fault_stations) == length(FAULT_STATION_KM) ||
        error("fault stations $(FAULT_STATION_KM) were not found on the grid")
    slip = zeros(length(stations))
    fault_slip = zeros(length(fault_stations))
    reset_vtk(dirname(pvd), pvd)
    csv = open_station_csv(csv_path)
    fault_csv = [open_fault_station_csv(fault_station_path(s.km)) for s in fault_stations]
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
        for (s, station) in enumerate(stations)
            sample = fault_sample(stokes, fields, j, station.k, L, V0)
            println(
                csv,
                "0.0,$(station.depth_m),0.0,$(sample.rate),$(sample.τ),$(sample.state),$(sample.Vp)",
            )
        end
        flush(csv)
        for (s, station) in enumerate(fault_stations)
            sample = fault_sample(stokes, fields, j, station.k, L, V0)
            write_fault_row!(fault_csv[s], 0.0, station, 0.0, sample)
            flush(fault_csv[s])
        end
        vtk_name = write_paraview(case, 0, 0.0, pvd, L, V0)
        println("vtk it=0 t=0.0 → $(vtk_name).vtr")
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
                    "$t,$(station.depth_m),$(slip[s]),$(sample.rate),$(sample.τ),$(sample.state),$(sample.Vp)",
                )
            end
            flush(csv)
            fault_line = String[]
            for (s, station) in enumerate(fault_stations)
                sample = fault_sample(stokes, fields, j, station.k, L, V0)
                fault_slip[s] += sample.Vj * dt
                write_fault_row!(fault_csv[s], t, station, fault_slip[s], sample)
                flush(fault_csv[s])
                push!(fault_line, "$(station.km)km τ=$(sample.τ) Pa V=$(sample.rate) m/s")
            end
            if it == 1 || rem(it, nstep_out) == 0
                vtk_name = write_paraview(case, it, t, pvd, L, V0)
                println("vtk it=$it t=$t → $(vtk_name).vtr")
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
    println("  paraview → $(pvd).pvd")
    println("  all depths → $csv_path")
    for station in fault_stations
        println("  fault $(station.km) km ($(station.depth_m) m) → $(fault_station_path(station.km))")
    end
    return (; case, out, t, it)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--run" in ARGS
        run_bp1()
    else
        smoke_bp1()
        println("connectivity only. Start the time loop with --run")
    end
end
