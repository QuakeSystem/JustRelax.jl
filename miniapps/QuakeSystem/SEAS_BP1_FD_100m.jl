#=
SEAS BP1-FD, viscosity-fold rate-and-state (not the eigenstrain split).

Grid is the same domain as the LaMEM reference (depth is LaMEM `z`,
fault-normal is LaMEM `y`; LaMEM `x = ±50 m` is the unused thickness), refined
to 600 × 600 cells. The fine zone is unchanged: 100 m from the surface to
−20 km (200 cells) and 100 m across ±500 m of the fault (10 cells). The other
400 depth cells and 295 cells on each side fill the same outer intervals, with
the LaMEM bias chosen so the cell that touches the fine zone is 100 m.

    nel_x = 400 200
    coord_x = -71700 -20000 0          # LaMEM coord_z
    nel_y = 295 10 295
    coord_y = -600000 -500 500 600000

The top 20 km and the ±500 m fault band are 100 m cells. `Wf = 100 m` is the
fault width in `V / (2 Wf)`. `Wf_height = 40 km` is the down-dip height of the
rate-and-state section. Below that, the two `Vx` nodes beside the fault
(`y = ±50 m`, the LaMEM VelBox faces) are held at `∓Vp/2`.

Side walls: the positive-y ghost is `+Vp/2` and the negative-y ghost is
`-Vp/2`, so the two plates move at half the plate rate in opposite directions.
The right edge `x = 0` is the free surface (`∂Vx/∂x = 0`). That face is a
boundary column the momentum kernel does not write; leaving the initial
velocity there shears the shallow fault on its own.

The fault is one vertex row at `y = 0` (phase 2). The host is elastic. The
initial fault stress is the SEAS prestress `τ0` from `a = 0.025`, the slip rate
is `Vp`, and `Ω(x)` is the state that returns that rate at the local `a(x)`.

`iterMax = 10000`, `iterMax_PH = 10`. A step that is still above the tolerance
is discarded and retried at half `dt`; time does not advance on that try.
Leave `isCUDA = false` here and switch it on for the device run. The arrays
are the same `PTArray` / `@zeros` allocations as the other DYREL miniapps.
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

const YEAR = 365.25 * 86400.0

@parallel_indices (i, j) function _seas_host_phase!(phases)
    @index phases[1, i, j] = 1.0
    @index phases[2, i, j] = 0.0
    return nothing
end

@parallel_indices (i, j) function _seas_fault_phase!(phases, j_fault)
    on = j == j_fault
    @index phases[1, i, j] = ifelse(on, 0.0, 1.0)
    @index phases[2, i, j] = ifelse(on, 1.0, 0.0)
    return nothing
end

"""
LaMEM `bias` (last/first cell size) whose fine end is `h_fine`.

`fine_at = :end` shrinks toward the segment end. `fine_at = :start` grows away
from the segment start. Both come from `endSz = 2 avg bias / (1 + bias)`.
"""
function lamem_bias(L, n; fine_at = :end, h_fine = 100.0)
    avg = L / n
    2 * avg > h_fine ||
        error("$n cells over $L m cannot meet a $h_fine m cell")
    b_end = h_fine / (2 * avg - h_fine)
    return fine_at === :end ? b_end : 1 / b_end
end

"""600 × 600 mesh. `x` is depth (LaMEM `z`), `y` is fault-normal (LaMEM `y`)."""
function seas_bp1_fd100m_mesh()
    L_deep = -20000.0 - (-71700.0)
    L_side = -500.0 - (-600000.0)
    return segmented_grid_2D(;
        nel_x = (400, 200),
        coord_x = (-71700.0, -20000.0, 0.0),
        bias_x = (lamem_bias(L_deep, 400; fine_at = :end), 1.0),
        nel_y = (295, 10, 295),
        coord_y = (-600000.0, -500.0, 500.0, 600000.0),
        bias_y = (
            lamem_bias(L_side, 295; fine_at = :end),
            1.0,
            lamem_bias(L_side, 295; fine_at = :start),
        ),
    )
end

function seas_rheology(; η, G, ν, ρ)
    creep = LinearViscous(; η = η)
    el = ConstantElasticity(; G = G, ν = ν)
    host = CompositeRheology((creep, el))
    return (
        SetMaterialParams(;
            Name = "Media",
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = host,
            Elasticity = el,
        ),
        SetMaterialParams(;
            Name = "Fault",
            Phase = 2,
            Density = ConstantDensity(; ρ = ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = host,
            Elasticity = el,
        ),
    )
end

function prepare_seas_vtk!(dir, pvd_name)
    mkpath(dir)
    for f in readdir(dir)
        if startswith(f, "step_") || endswith(f, ".pvd") || endswith(f, ".vtu") ||
                endswith(f, ".vtr") || endswith(f, ".csv")
            rm(joinpath(dir, f); force = true)
        end
    end
    return joinpath(dir, pvd_name)
end

function write_seas_vtk!(vtk_dir, pvd, it, t_s, xvi, xci, stokes, rsf_fields, Vx_v, Vy_v)
    velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
    data_v = (;
        Vx = Array(Vx_v),
        Vy = Array(Vy_v),
        τxy = Array(stokes.τ.xy),
        Vp = Array(rsf_fields.Vp_v),
        Omega = Array(rsf_fields.Ωv),
        a = Array(rsf_fields.a_eff_v),
    )
    data_c = (; P = Array(stokes.P), η = Array(stokes.viscosity.η))
    path = joinpath(vtk_dir, "step_" * lpad(string(it), 6, '0'))
    save_vtk(path, xvi, xci, data_v, data_c, (data_v.Vx, data_v.Vy); t = t_s, pvd = pvd)
    return nothing
end

"""Indices of the LaMEM VelBox faces: `Vx` at `y = ±50 m`, and `x ≤ −Wf_height`."""
function fault_plate_indices(xVx, yVx, Wf_height)
    j_neg = argmin(abs.(yVx .+ 50.0))
    j_pos = argmin(abs.(yVx .- 50.0))
    abs(yVx[j_neg] + 50.0) < 1.0e-6 ||
        error("no Vx node at y = −50 m (closest $(yVx[j_neg]))")
    abs(yVx[j_pos] - 50.0) < 1.0e-6 ||
        error("no Vx node at y = +50 m (closest $(yVx[j_pos]))")
    i2 = findlast(≤(-Wf_height), xVx)
    isnothing(i2) && error("no Vx node at or below −$Wf_height m")
    return (; i1 = 1, i2 = i2, j_neg = j_neg, j_pos = j_pos)
end

function check_bp1_mesh(xv, yv)
    j_fault = argmin(abs.(yv))
    yv[j_fault] ≈ 0.0 || error("fault vertex y = $(yv[j_fault]) is not 0")
    dy_lo = yv[j_fault] - yv[j_fault - 1]
    dy_hi = yv[j_fault + 1] - yv[j_fault]
    (dy_lo ≈ 100.0 && dy_hi ≈ 100.0) ||
        error("fault is not between two 100 m cells (dy = $dy_lo, $dy_hi)")
    i20 = findfirst(≈(-20000.0), xv)
    isnothing(i20) && error("no vertex at x = −20 km")
    all(diff(xv[i20:end]) .≈ 100.0) || error("the top 20 km is not uniform 100 m")
    # The outer segment must meet the 100 m zone, not jump.
    dx_deep = xv[i20] - xv[i20 - 1]
    j_neg = argmin(abs.(yv .+ 500.0))
    j_pos = argmin(abs.(yv .- 500.0))
    dy_side = yv[j_neg] - yv[j_neg - 1]
    dy_side_pos = yv[j_pos + 1] - yv[j_pos]
    abs(dx_deep - 100.0) < 1.0 ||
        error("cell below −20 km is $dx_deep m; the deep bias does not meet 100 m")
    abs(dy_side - 100.0) < 1.0 ||
        error("cell outside −500 m is $dy_side m; the side bias does not meet 100 m")
    abs(dy_side_pos - 100.0) < 1.0 ||
        error("cell outside +500 m is $dy_side_pos m; the side bias does not meet 100 m")
    return (; j_fault, dy_lo, dy_hi, dx_deep, dy_side)
end

function main(
        igg;
        xvi,
        Wf = 100.0,
        Wf_height = 40.0e3,
        Vp = 1.0e-9,
        η0 = 5.0e26,
        G0 = 3.2038e10,
        ν = 0.25,
        ρ = 2670.0,
        σ = 50.0e6,
        a_x = (-18.0e3, -15.0e3),
        a_val = (0.025, 0.01),
        b = 0.015,
        μ₀ = 0.6,
        V₀ = 1.0e-6,
        L = 0.008,
        dt = 2522.0,
        dt_min = 5.0e-3,
        dt_max = 1.0e9,
        nsteps = 245_000,
        max_time = 3.0e10,
        iterMax = 1.0e4,
        iterMax_PH = 10,
        vtk_dir = nothing,
        pvd_name = "seas_bp1_fd_100m",
        vtk_every = 10,
    )
    ni = (length(xvi[1]) - 1, length(xvi[2]) - 1)
    grid = Geometry(PTArray(backend), xvi...)
    xv = Array(grid.xvi[1])
    yv = Array(grid.xvi[2])
    mesh_ok = check_bp1_mesh(xv, yv)
    j_fault = mesh_ok.j_fault

    rheology = seas_rheology(; η = η0, G = G0, ν = ν, ρ = ρ)
    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    @parallel (@idx size(phase_ratios.center)) _seas_host_phase!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _seas_fault_phase!(phase_ratios.vertex, j_fault)

    a_max = maximum(a_val)
    τ0 = eigenstrain_steady_stress(a_max, σ, μ₀, b, V₀, Vp)
    rsf_nt = (
        enabled = true,
        affect_stokes = true,
        loc = :vertex,
        V0 = V₀,
        dt_min = dt_min,
        dt_max = dt_max,
        dt_rsf_switch = 1.0e12,
        G = G0,
        ν = ν,
        Phase2 = (
            a_rsf = a_val[1],
            b_rsf = b,
            a_rsf_val = a_val,
            a_rsf_x = a_x,
            mu0_rsf = μ₀,
            D_rs = L,
            Wf = Wf,
            state_rsf_init = 0.0,
            λ = 0.0,
            C = 0.0,
        ),
    )
    di_min = ntuple(d -> minimum(Array(grid.di.vertex[d])), 2)
    ctrl = build_rate_state_controller(rsf_nt; nphases = length(rheology), di = di_min)
    rsf_fields = RateStateArrays(backend, ni; loc = :vertex)
    init_rate_state_fields!(rsf_fields, ctrl, phase_ratios, grid.xci, grid.xvi)
    rsf = (; ctrl = ctrl, fields = rsf_fields)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = false),
        periodic = (left = false, right = false, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, τ0)
    fill!(stokes.τ_o.xy, τ0)

    xVx = Array(grid.xi_vel[1][1])
    yVx = Array(grid.xi_vel[1][2])
    plates = fault_plate_indices(xVx, yVx, Wf_height)
    half = 0.5 * Vp
    bcvx = (;
        free_surface = :right,
        top = half,
        bot = -half,
        fault_plates = (; plates.i1, plates.i2, plates.j_neg, plates.j_pos, V_neg = -half, V_pos = half),
    )
    Vx0 = [y ≥ 0.0 ? half : -half for _ in xVx, y in yVx]
    stokes.V.Vx .= PTArray(backend)(Vx0)
    fill!(stokes.V.Vy, 0.0)
    JustRelax.apply_boundary_vx!(stokes.V.Vx, bcvx)
    stokes.V0.Vx .= stokes.V.Vx
    stokes.V0.Vy .= stokes.V.Vy
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    # init_seas_tau0: one stress, local a(x), rate Vp. Overwrites state_rsf_init.
    Ωh = Array(rsf_fields.Ωv)
    ah = Array(rsf_fields.a_eff_v)
    for i in axes(Ωh, 1)
        Ωh[i, j_fault] = eigenstrain_ψ_for_slip(ah[i, j_fault], b, μ₀, V₀, Vp, τ0, σ)
    end
    copyto!(rsf_fields.Ωv, Ωh)
    copyto!(rsf_fields.Ω_old_v, Ωh)

    i125 = argmin(abs.(xv .+ 12.5e3))
    stations = NamedTuple[]
    for depth in (0.0, 10.0e3, 12.5e3, 20.0e3)
        i = argmin(abs.(xv .+ depth))
        push!(stations, (; depth = depth, i = i, x = xv[i]))
    end
    println(
        "BP1-FD viscosity fold  ni=$ni  fault j=$j_fault  " *
            "dy=($(mesh_ok.dy_lo), $(mesh_ok.dy_hi)) m  " *
            "cell below −20 km = $(round(mesh_ok.dx_deep; digits = 2)) m",
    )
    println(
        "τ0 = $(τ0 / 1.0e6) MPa   a(−12.5 km)=$(ah[i125, j_fault])  " *
            "a(surface)=$(ah[stations[1].i, j_fault])  " *
            "prescribed Vx through x=$(xVx[plates.i2]) m  " *
            "walls ±$(half) m/s",
    )
    ah[stations[1].i, j_fault] ≈ 0.01 || error("surface a is not 0.010")
    ah[argmin(abs.(xv .+ 20.0e3)), j_fault] ≈ 0.025 || error("a at −20 km is not 0.025")
    flush(stdout)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6, CFL = 0.9)
    ρg = @zeros(ni...), @zeros(ni...)
    xvi_h = Array.(grid.xvi)
    xci_h = Array.(grid.xci)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    pvd = nothing
    hist = nothing
    probe = nothing
    probe125 = nothing
    if !isnothing(vtk_dir)
        pvd = prepare_seas_vtk!(vtk_dir, pvd_name)
        hist = open(joinpath(vtk_dir, "history.csv"), "w")
        println(hist, "step,t_year,t_s,iter,rejected,dt,dt_rsf,Vp_max,tau_12.5km,V_12.5km,wall_s")
        probe = open(joinpath(vtk_dir, "fault_stations.csv"), "w")
        println(probe, "Time,Dt,depth_km,x_m,Tau,V,Psi,log10_theta")
        # One row per time. ParaView reads this as a table (plot Tau against Time)
        # or as a point after Table to Points on X, Y, Z.
        probe125 = open(joinpath(vtk_dir, "fault_12.5km.csv"), "w")
        println(probe125, "Time,Dt,X,Y,Z,Tau,Vp,Omega,log10_theta,Vx,Vy")
        flush(hist)
        flush(probe)
        flush(probe125)
        println("probe 12.5 km: $(joinpath(vtk_dir, "fault_12.5km.csv"))")
    end

    function write_stations!(io, t_now, dt_now, τ_row, V_row, ψ_row)
        isnothing(io) && return nothing
        for s in stations
            println(
                io,
                "$t_now,$dt_now,$(s.depth / 1.0e3),$(s.x),$(τ_row[s.i]),$(V_row[s.i]),$(ψ_row[s.i]),$(ψ_row[s.i] / log(10) + log10(L / V₀))",
            )
        end
        flush(io)
        return nothing
    end

    function write_probe_12_5!(io, t_now, dt_now, τ, Vp_pt, ψ, Vx_pt, Vy_pt)
        isnothing(io) && return nothing
        println(
            io,
            "$t_now,$dt_now,$(xv[i125]),0.0,0.0,$τ,$Vp_pt,$ψ,$(ψ / log(10) + log10(L / V₀)),$Vx_pt,$Vy_pt",
        )
        flush(io)
        return nothing
    end

    function fault_vertex_velocity()
        velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
        return Array(Vx_v)[i125, j_fault], Array(Vy_v)[i125, j_fault]
    end

    t = 0.0
    it = 0
    rejects = 0
    try
        if !isnothing(pvd)
            write_seas_vtk!(vtk_dir, pvd, 0, 0.0, xvi_h, xci_h, stokes, rsf_fields, Vx_v, Vy_v)
            println("ParaView: $(pvd).pvd   (time axis is seconds)")
            flush(stdout)
        end
        vx0, vy0 = fault_vertex_velocity()
        write_probe_12_5!(probe125, 0.0, 0.0, τ0, Vp, Ωh[i125, j_fault], vx0, vy0)
        while it < nsteps && t < max_time
            wall0 = time()
            args = (; args..., dt = dt)
            out = solve_DYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
                inertia = true,
                ρ_inertia = ρ,
                verbose_PH = false,
                verbose_DR = false,
                iterMax = iterMax,
                iterMax_PH = iterMax_PH,
                nout = 100,
                total_iterMax = iterMax * iterMax_PH,
                rel_drop = 1.0e-3,
                linear_viscosity = true,
                update_material = false,
                rsf = rsf,
                boundary_vx = bcvx,
            )
            JustRelax.apply_boundary_vx!(stokes.V.Vx, bcvx)
            # A discarded step returns dt/2 with the three rate-and-state limits left at Inf.
            rejected = !isfinite(out.dt_h) && isfinite(out.dt_rsf)
            τ_row = Array(stokes.τ.xy)[:, j_fault]
            V_row = Array(rsf_fields.Vp_v)[:, j_fault]
            ψ_row = Array(rsf_fields.Ωv)[:, j_fault]
            all(isfinite, τ_row) && all(isfinite, V_row) ||
                error("non-finite fault state at it=$it t=$t s")
            wall = time() - wall0
            if rejected
                rejects += 1
                dt_next = out.dt_rsf
                dt_next ≥ dt_min || error(
                    "rejected step asked for dt=$dt_next s, below dt_min=$dt_min s " *
                        "(iter=$(out.iter), τ(12.5 km)=$(τ_row[i125] / 1.0e6) MPa)",
                )
                @printf(
                    "reject %d  t=%9.4f yr  iter=%d  dt %.3e → %.3e s  12.5km τ=%.4f MPa  wall=%.2f s\n",
                    rejects, t / YEAR, out.iter, dt, dt_next, τ_row[i125] / 1.0e6, wall,
                )
                flush(stdout)
                dt = dt_next
                continue
            end
            it += 1
            t += dt
            dt_next = isfinite(out.dt_rsf) ? clamp(out.dt_rsf, dt_min, dt_max) : dt
            @printf(
                "it=%d  t=%9.4f yr  iter=%d  Vp_max=%.3e  dt=%.3e s  dt_rsf=%.3e s  wall=%.2f s  12.5km τ=%.4f MPa V=%.3e\n",
                it, t / YEAR, out.iter, out.Vp_max, dt, out.dt_rsf, wall, τ_row[i125] / 1.0e6, V_row[i125],
            )
            flush(stdout)
            if !isnothing(hist)
                println(
                    hist,
                    "$it,$(t / YEAR),$t,$(out.iter),0,$dt,$(out.dt_rsf),$(out.Vp_max),$(τ_row[i125]),$(V_row[i125]),$wall",
                )
                flush(hist)
            end
            write_stations!(probe, t, dt, τ_row, V_row, ψ_row)
            vx_pt, vy_pt = fault_vertex_velocity()
            write_probe_12_5!(probe125, t, dt, τ_row[i125], V_row[i125], ψ_row[i125], vx_pt, vy_pt)
            if !isnothing(pvd) && (it == 1 || rem(it, vtk_every) == 0)
                write_seas_vtk!(vtk_dir, pvd, it, t, xvi_h, xci_h, stokes, rsf_fields, Vx_v, Vy_v)
            end
            dt = dt_next
        end
        println("stopped at t=$(t / YEAR) yr after $it steps ($rejects rejected)")
        flush(stdout)
    finally
        isnothing(hist) || close(hist)
        isnothing(probe) || close(probe)
        isnothing(probe125) || close(probe125)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    mesh = seas_bp1_fd100m_mesh()
    xv, yv = mesh.xvi
    info = check_bp1_mesh(xv, yv)
    mesh.ni == (600, 600) || error("expected 600×600 cells, got $(mesh.ni)")
    dx, dy = diff(xv), diff(yv)
    jump(h) = maximum(max.(h[2:end] ./ h[1:(end - 1)], h[1:(end - 1)] ./ h[2:end]))
    println(
        "ni = $(mesh.ni)   Δx ∈ $(extrema(dx)) m   Δy ∈ $(extrema(dy)) m   " *
            "max adjacent ratio $(round(max(jump(dx), jump(dy)); digits = 3))",
    )
    println(
        "fault j=$(info.j_fault)  dy=($(info.dy_lo), $(info.dy_hi)) m  " *
            "cell below −20 km = $(info.dx_deep) m  cell outside ±500 m = $(info.dy_side) m",
    )
    flush(stdout)
    nx, ny = mesh.ni
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    main(
        igg;
        xvi = mesh.xvi,
        vtk_dir = joinpath(@__DIR__, "VTK_SEAS_BP1_FD_100m", "vtk"),
    )
end
