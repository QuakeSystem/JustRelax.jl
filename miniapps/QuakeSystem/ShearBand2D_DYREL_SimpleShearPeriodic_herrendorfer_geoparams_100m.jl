#=
Herrendörfer viscoelastic simple shear with GeoParams rate-and-state friction
on a 100 m grid (CUDA).

Same physics as the 500 m miniapp, with:
    nel_x = 1500          → Δx = 100 m on [-75 km, 75 km]
    nel_y = (140, 140, 140)
    Wf    = 100 m

Requires CUDA. Prefer 8 Julia threads for host-side work:

    julia -t 8 --project=. miniapps/QuakeSystem/ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer_geoparams_100m.jl

Smoke connectivity (no DYREL loop):

    julia -t 8 --project=. miniapps/QuakeSystem/ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer_geoparams_100m.jl --smoke
=#
const isCUDA = true

using CUDA

using JustRelax, JustRelax.JustRelax2D

const backend = JustRelax.CUDABackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)

using JustPIC
const backend_JP = CUDA.CUDABackend

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
        nel_x = 1500,
        coord_x = (-75.0e3, 75.0e3),
        nel_y = (140, 140, 140),
        coord_y = (-150.0e3, -76.0e3, -74.0e3, 0.0),
        fault = (; x = (-75.0e3, 75.0e3), y = (-75.2e3, -74.8e3)),
        V_top = 4.0e-9,
        η0 = 5.0e26,
        G0 = 3.0e10,
        ν = 0.25,
        P0 = 5.0e6,
        Wf = 100.0,   # LaMEM fault width [m]
        dt = 500.0,
        nsteps = 103000,
        t_max = 1.0e9,
        dt_min = 1.0e-1,
        dt_max = 1.0e7,
        dt_rsf_switch = 1.0e9,
        inertia = true,
        ρ_inertia = nothing,
        probe = (; x = -50.0e3, z = -75.0e3),
        csv_path = joinpath(@__DIR__, "VTK", "probe_100m_x-50000_z-75000.csv"),
        print_every = 10,
    )
    CUDA.functional() || error("CUDA is not functional on this machine.")

    mesh = segmented_grid_2D(;
        nel_x = nel_x, coord_x = coord_x,
        nel_y = nel_y, coord_y = coord_y,
    )
    (; xvi, ni, li, origin) = mesh
    nx, ny = ni
    lx, ly = li
    Δx = lx / nx
    isapprox(Δx, 100.0; rtol = 1.0e-12) ||
        @warn "Expected Δx ≈ 100 m, got Δx = $Δx m (nel_x=$nel_x, lx=$lx)"

    staggered_grid, ph_vertex, _T = simple_shear_2D(xvi, 1250.0, fault, (; do_vtk = false))
    grid = Geometry(PTArray(backend), xvi...)
    (; xci, xvi) = grid

    rheology = init_rheology_rate_state(; η = η0, G = G0, ν = ν, V₀ = V_top, Wf = Wf)
    rsf_law = GeoParams.rate_state_element(rheology[2].CompositeRheology[1])
    float(rsf_law.D.val) ≈ Wf ||
        error("GeoParams RateStateFriction D=$(rsf_law.D.val) does not match Wf=$Wf")

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
    println("threads=$(Threads.nthreads())  CUDA device=$(CUDA.name(CUDA.device()))")

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
        "GeoParams RSF Herrendörfer 100 m (CUDA): ni=$ni, Δx=$(Δx) m, Wf=$Wf m, " *
            "dt=$dt, nsteps=$nsteps, t_max=$t_max, inertia=$inertia",
    )
    t = 0.0
    it = 0
    out = nothing
    try
        while it < nsteps && t < t_max
            out = solve_DYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
                kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 50.0e3,
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

            Vx_p = Float64(stokes.V.Vx[ip, jp])
            Vy_p = Float64(stokes.V.Vy[min(ip, size(stokes.V.Vy, 1)), min(jp, size(stokes.V.Vy, 2))])
            τxy_p = Float64(stokes.τ.xy[ip, jp])
            ηv_p = Float64(stokes.viscosity.ηv[ip, jp])
            Vp_p = Float64(fields.Vp_v[ip, jp])
            Ω_p = Float64(fields.Ωv[ip, jp])
            a_p = Float64(fields.a_eff_v[ip, jp])
            b_p = Float64(fields.b_eff_v[ip, jp])
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

"""
Build mesh, rheology, GPU Stokes arrays and RSF fields. No `solve_DYREL!`.
"""
function smoke_connectivity(;
        nel_x = 1500,
        nel_y = (140, 140, 140),
        Wf = 100.0,
        coord_x = (-75.0e3, 75.0e3),
        coord_y = (-150.0e3, -76.0e3, -74.0e3, 0.0),
        fault = (; x = (-75.0e3, 75.0e3), y = (-75.2e3, -74.8e3)),
    )
    CUDA.functional() || error("CUDA is not functional on this machine.")
    Threads.nthreads() >= 8 ||
        @warn "Prefer julia -t 8; currently Threads.nthreads()=$(Threads.nthreads())"

    mesh = segmented_grid_2D(; nel_x = nel_x, coord_x = coord_x, nel_y = nel_y, coord_y = coord_y)
    (; xvi, ni, li) = mesh
    Δx = li[1] / ni[1]
    dy = diff(Array(xvi[2]))
    # Middle y-segment is cells (nel_y[1]+1):(nel_y[1]+nel_y[2])
    dy_mid = dy[(nel_y[1] + 1):(nel_y[1] + nel_y[2])]
    println(
        "smoke: ni=$ni  Δx=$(Δx) m  Δy_mid∈$(extrema(dy_mid)) m  " *
            "threads=$(Threads.nthreads())",
    )
    println("smoke: CUDA $(CUDA.name(CUDA.device()))")
    isapprox(Δx, 100.0; rtol = 1.0e-12) || error("Δx=$Δx, expected 100 m")
    ni == (1500, 420) || error("ni=$ni, expected (1500, 420)")

    rheology = init_rheology_rate_state(; Wf = Wf)
    for ip in 1:2
        r = GeoParams.rate_state_element(rheology[ip].CompositeRheology[1])
        float(r.D.val) ≈ Wf || error("phase $ip D=$(r.D.val) ≠ Wf=$Wf")
    end
    println("smoke: Wf = $(Wf) m on both phases")

    _, ph_vertex, _ = simple_shear_2D(xvi, 1250.0, fault, (; do_vtk = false))
    grid = Geometry(PTArray(backend), xvi...)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phase_ratios_from_grid!(phase_ratios, ph_vertex, length(rheology))
    fields = RateStateArrays(backend, ni; loc = :vertex)
    init_geoparams_rate_state!(
        fields, rheology, phase_ratios, (40.0, -1.0), grid.xvi;
        xci = grid.xci,
        b_profiles = (nothing, (B_FAULT_X, B_FAULT_VAL)),
    )
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, 5.0e6)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = 500.0)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (1.0e6, 5.0e26))
    apply_geoparams_rsf_viscosity!(
        stokes, fields, phase_ratios, rheology, 500.0; viscosity_cutoff = (1.0e6, 5.0e26),
    )
    CUDA.synchronize()
    ηv_min = minimum(Array(stokes.viscosity.ηv))
    mask_max = maximum(Array(fields.rsf_mask_v))
    isfinite(ηv_min) && ηv_min > 0 || error("ηv not finite after GPU RSF apply")
    mask_max ≈ 1.0 || error("rsf_mask_v empty on GPU")
    println("smoke: GPU arrays OK  min(ηv)=$ηv_min  max(rsf_mask_v)=$mask_max")
    println("smoke: connectivity OK")
    return (; ni, Δx, Wf, ηv_min)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--smoke" in ARGS
        smoke_connectivity()
    else
        nel_x = 1500
        nel_y = (140, 140, 140)
        nx = nel_x
        ny = sum(nel_y)
        igg = if !(JustRelax.MPI.Initialized())
            IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
        else
            igg
        end
        @time main(
            igg;
            nel_x = nel_x,
            nel_y = nel_y,
            Wf = 100.0,
            nsteps = 103000,
            t_max = 1.0e9,
            dt = 500.0,
            inertia = true,
            probe = (; x = -50.0e3, z = -75.0e3),
            csv_path = joinpath(@__DIR__, "VTK", "probe_100m_x-50000_z-75000.csv"),
        )
    end
end
