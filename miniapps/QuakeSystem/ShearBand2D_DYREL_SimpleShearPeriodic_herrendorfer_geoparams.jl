#=
Herrendörfer viscoelastic simple shear with GeoParams rate-and-state friction.

Series CompositeRheology (linear dashpot, RateStateFriction, elasticity).
`solve_DYREL!` uses `gp_rsf` (RateStateArrays). Ω is frozen during iterations and
advanced after the step from the frictional strain rate only.

Default run: 300×24 mesh, up to 103000 steps or t = 1e9 s. Writes a ParaView-readable
CSV time series at the velocity-strengthening probe (x, z) = (−50 km, −75 km).
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D

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
include(joinpath(@__DIR__, "simple_shear_rheology.jl"))

const B_FAULT_X = (-47000.0, -43000.0, 33000.0, 37000.0)
const B_FAULT_VAL = (0.001, 0.017, 0.017, 0.001)

"""Nearest vertex indices to (x_probe, z_probe) on the staggered grid."""
function nearest_vertex_indices(xvi, x_probe, z_probe)
    x = Array(xvi[1])
    z = Array(xvi[2])
    i = argmin(abs.(x .- x_probe))
    j = argmin(abs.(z .- z_probe))
    return i, j, x[i], z[j]
end

"""
Open a ParaView CSV for a fixed probe. Columns are suitable for Delimited Text /
TableToPoints (X,Y,Z plus time and fields).
"""
function open_probe_csv(path, x, y, z)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, "time,X,Y,Z,Vx,Vy,tau_xy,Vp,Omega,a_eff,b_eff,eta_v,P")
    flush(io)
    println("probe CSV → $path  at ($(x), $(y), $(z))")
    return io
end

function write_probe_csv!(io, t, x, y, z, Vx, Vy, τxy, Vp, Ω, a, b, ηv, P)
    println(
        io,
        join(
            (
                t, x, y, z, Vx, Vy, τxy, Vp, Ω, a, b, ηv, P,
            ),
            ',',
        ),
    )
    flush(io)
    return nothing
end

function main(
        igg;
        nel_x = 300,
        coord_x = (-75.0e3, 75.0e3),
        nel_y = (10, 4, 10),
        coord_y = (-150.0e3, -76.0e3, -74.0e3, 0.0),
        fault = (; x = (-75.0e3, 75.0e3), y = (-75.2e3, -74.8e3)),
        V_top = 4.0e-9,
        η0 = 5.0e26,
        G0 = 3.0e10,
        ν = 0.25,
        P0 = 5.0e6,
        Wf = 500.0,   # LaMEM fault width [m]
        dt = 500.0,
        nsteps = 103000,
        t_max = 1.0e9,
        dt_min = 1.0e-1,
        dt_max = 1.0e7,
        dt_rsf_switch = 1.0e9,
        inertia = true,
        ρ_inertia = nothing,
        probe = (; x = -50.0e3, z = -75.0e3),
        csv_path = joinpath(@__DIR__, "VTK", "probe_x-50000_z-75000.csv"),
        print_every = 1,
    )
    mesh = segmented_grid_2D(;
        nel_x = nel_x, coord_x = coord_x,
        nel_y = nel_y, coord_y = coord_y,
    )
    (; xvi, ni, li, origin) = mesh
    nx, ny = ni
    lx, ly = li

    staggered_grid, ph_vertex, _T = simple_shear_2D(xvi, 1250.0, fault, (; do_vtk = false))
    grid = Geometry(PTArray(backend), xvi...)
    (; xci, xvi) = grid

    rheology = init_rheology_rate_state(; η = η0, G = G0, ν = ν, V₀ = V_top, Wf = Wf)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phase_ratios_from_grid!(phase_ratios, ph_vertex, length(rheology))

    fields = RateStateArrays(backend, ni; loc = :vertex)
    init_geoparams_rate_state!(
        fields, rheology, phase_ratios, (40.0, -1.0), xvi;
        xci = xci,
        b_profiles = (nothing, (B_FAULT_X, B_FAULT_VAL)),
    )

    ip, jp, x_p, z_p = nearest_vertex_indices(xvi, probe.x, probe.z)
    csv_io = open_probe_csv(csv_path, x_p, z_p, 0.0)
    println("probe vertex indices (i,j)=($ip,$jp); requested ($(probe.x), $(probe.z))")

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, P0)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    viscosity_cutoff = (1.0e6, 5.0e26)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    xvi_h = Array.(xvi)
    yVx = Array(grid.xi_vel[1][2])
    ybot, ytop = origin[2], origin[2] + ly
    stokes.V.Vx .= PTArray(backend)([
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi_h[1], y in yVx
        ])
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)

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
        fill!(stokes.V0.Vx, 0.0)
        fill!(stokes.V0.Vy, 0.0)
    end

    println(
        "GeoParams RSF Herrendörfer: ni=$ni, Wf=$Wf m, dt=$dt, nsteps=$nsteps, " *
            "t_max=$t_max, inertia=$inertia",
    )
    t = 0.0
    it = 0
    out = nothing
    try
        while it < nsteps && t < t_max
            out = solve_DYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
                kwargs = (;
                    verbose_PH = true,
                    verbose_DR = false,
                    iterMax = 50.0e2,
                    nout = 50,
                    rel_drop = 1.0e-4,
                    linear_viscosity = true,
                    viscosity_cutoff = viscosity_cutoff,
                    gp_rsf = fields,
                    inertia = inertia,
                    ρ_inertia = ρ_for_inertia,
                ),
            )
            @views stokes.V.Vx[:, end] .= V_top
            it += 1
            t += dt

            # Vertex probe: τ.xy / ηv / Vp_v / Ωv live on shear nodes.
            Vx_p = Float64(stokes.V.Vx[ip, jp])
            Vy_p = Float64(stokes.V.Vy[min(ip, size(stokes.V.Vy, 1)), min(jp, size(stokes.V.Vy, 2))])
            τxy_p = Float64(stokes.τ.xy[ip, jp])
            ηv_p = Float64(stokes.viscosity.ηv[ip, jp])
            Vp_p = Float64(fields.Vp_v[ip, jp])
            Ω_p = Float64(fields.Ωv[ip, jp])
            a_p = Float64(fields.a_eff_v[ip, jp])
            b_p = Float64(fields.b_eff_v[ip, jp])
            # Pressure at the four surrounding cell centers.
            nxP, nyP = size(stokes.P)
            i0 = clamp(ip - 1, 1, nxP)
            ic = clamp(ip, 1, nxP)
            j0 = clamp(jp - 1, 1, nyP)
            jc = clamp(jp, 1, nyP)
            P_p = 0.25 * (
                Float64(stokes.P[i0, j0]) + Float64(stokes.P[ic, jc]) +
                    Float64(stokes.P[i0, jc]) + Float64(stokes.P[ic, j0])
            )
            write_probe_csv!(csv_io, t, x_p, z_p, 0.0, Vx_p, Vy_p, τxy_p, Vp_p, Ω_p, a_p, b_p, ηv_p, P_p)

            if it == 1 || rem(it, print_every) == 0 || t >= t_max || it >= nsteps
                println(
                    "it=$it  t=$t  max|τxy|=$(maximum(abs, Array(stokes.τ.xy)))  " *
                        "Vp_max=$(out.Vp_max)  dt=$dt  dt_rsf=$(out.dt_rsf)  " *
                        "probe Vp=$Vp_p  Ω=$Ω_p  b=$b_p",
                )
            end

            if isfinite(out.dt_rsf) && out.dt_rsf < dt_rsf_switch
                dt = clamp(out.dt_rsf, dt_min, dt_max)
                args = (; args..., dt = dt)
            end
        end
    finally
        close(csv_io)
    end
    println("finished it=$it t=$t → $csv_path")
    return (; stokes, fields, rheology, phase_ratios, grid, out, t, it, lx, ly, csv_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    nel_x = 300
    nel_y = (10, 4, 10)
    nx = nel_x
    ny = sum(nel_y)
    igg = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
    else
        igg
    end
    @time main(
        igg;
        nsteps = 103000,
        t_max = 1.0e10,
        dt = 500.0,
        inertia = true,
        probe = (; x = -50.0e3, z = -75.0e3),
        csv_path = joinpath(@__DIR__, "VTK", "probe_x-50000_z-75000.csv"),
    )
end
