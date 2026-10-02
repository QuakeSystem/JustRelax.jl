#=
Herrendörfer-style viscoelastic simple shear (phase 1).

- Left/right: periodic velocity BCs (seam residual; no IGG `periodx`)
- Bottom: no-slip; top: prescribed plate velocity `V_top` [m/s]
- Rheology: linear viscoelastic background + softer elastic sphere (no plasticity / RSF yet)
- Output: ParaView VTK + PVD with RSF placeholder fields set to zero

Single MPI rank only.
=#
const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg; Pkg.activate("miniapps")

const backend = @static if isCUDA
    JustRelax.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

using GeoParams

import JustPIC.GridGeometryUtils as GGU

# Analytic Maxwell shear stress for Couette flow with engineering shear rate
# ε̇ = (V_top - V_bot) / ly (= ∂Vx/∂y). With the tensor strain rate εxy = ε̇/2,
# τxy = 2G εxy t = G ε̇ t at early (elastic) times.
solution(ε̇, t, G, η) = ε̇ * η * (1 - exp(-G * t / η))

function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        p = GGU.Point(x, y)
        if GGU.inside(p, circle)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)
    return nothing
end

function phase_map(phase_ratios_loc)
    # CellArray → Array of SVector{nphases}; argmax → phase index 1..nphases
    return [argmax(p) for p in Array(phase_ratios_loc)]
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
        VTK, it, t, xvi, xci, stokes, phase_ratios, Vx_v, Vy_v, zeros_c, τ_analytic
    )
    VTK.do_vtk || return nothing
    (it == 1 || rem(it, VTK.vtk_every) == 0) || return nothing

    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    phase_vertex = phase_map(phase_ratios.vertex)
    # Homogeneous Phase-1 Maxwell τxy as a constant field for elastic-loading checks
    τxy_analytic = fill(τ_analytic, size(zeros_c))

    data_v = (;
        Vx = Array(Vx_v),
        Vy = Array(Vy_v),
        phase = phase_vertex,
    )
    data_c = (;
        P = Array(stokes.P),
        T = zeros_c,
        η = Array(stokes.viscosity.η),
        τII = Array(stokes.τ.II),
        εII = Array(stokes.ε.II),
        RP = Array(stokes.R.RP),
        τxy_analytic = τxy_analytic,
        # RSF placeholders for the next phase
        Ω = zeros_c,
        Vp = zeros_c,
        log10_Vp = zeros_c,
        τ_rsf = zeros_c,
        a_eff = zeros_c,
        b_eff = zeros_c,
    )
    velocity_v = (Array(Vx_v), Array(Vy_v))
    path_vtk = joinpath(VTK.vtk_dir, "vtk_" * lpad("$it", 6, "0"))
    save_vtk(
        path_vtk,
        xvi,
        xci,
        data_v,
        data_c,
        velocity_v;
        t = t,
        pvd = joinpath(VTK.vtk_dir, VTK.pvd_name),
    )
    println("Saved VTK → $(joinpath(VTK.vtk_dir, VTK.pvd_name)).pvd  (it=$it, t=$t)")
    return nothing
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(
        igg;
        nx = 64,
        ny = 64,
        lx = 150.0e3,
        ly = 150.0e3,
        origin = (-75.0e3, -75.0e3),
        V_top = 4.0e-9,
        η0 = 1.0e23,
        G0 = 3.0e10,
        ν = 0.25,
        dt = 500.0,
        nsteps = 10,
        radius_frac = 0.1,
        VTK = nothing,
    )

    # Physical domain ------------------------------------
    ni = nx, ny
    li = lx, ly
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid
    # Engineering shear rate for fixed bottom + moving top
    εbg = V_top / ly

    # Physical properties (Herrendörfer / LaMEM-style Maxwell buffer) -------------
    Gi = G0 / 2
    el_bg = ConstantElasticity(; G = G0, ν = ν)
    el_inc = ConstantElasticity(; G = Gi, ν = ν)
    visc = LinearViscous(; η = η0)

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg)),
            Elasticity = el_bg,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc)),
            Elasticity = el_inc,
        ),
    )

    # Phases: viscoelastic background + softer sphere ---------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    cx = origin[1] + lx / 2
    cy = origin[2] + ly / 2
    radius = radius_frac * min(lx, ly)
    circle = GGU.Circle((cx, cy), radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # Couette BCs (Herrendörfer): no-slip bottom, prescribed Vx on top, periodic in x.
    # Top has no free_slip/no_slip/periodic flag so `flow_bcs!` leaves `V_top` untouched.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )

    stokes = StokesArrays(backend, ni, flow_bcs)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # η ≫ G·dt keeps the Maxwell buffer quasi-elastic (η_ve ≈ G·dt)
    viscosity_cutoff = (1.0e10, 1.0e28)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # Linear Couette initial guess from fixed bottom (0) to V_top; bottom is then
    # enforced by no-slip, top ghost row keeps V_top (solver does not overwrite it).
    yVx = grid.xi_vel[1][2]
    ybot, ytop = origin[2], origin[2] + ly
    stokes.V.Vx .= PTArray(backend)([
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi[1], y in yVx
        ])
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # IO -----------------------------------------------------------------------
    isnothing(VTK) && (VTK = (;
            do_vtk = true,
            folder = joinpath(@__DIR__, "VTK"),
            vtk_dir = joinpath(@__DIR__, "VTK", "vtk"),
            pvd_name = "simple_shear_herrendorfer",
            vtk_every = 1,
        ))
    prepare_vtk!(VTK)

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    zeros_c = zeros(nx, ny)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    t, it = 0.0, 0
    τII = [0.0]
    sol = [0.0]
    ttot = [0.0]

    for _ in 1:nsteps
        solve_DYREL!(
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
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 10,
                rel_drop = 1.0e-2,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1,
                linear_viscosity = true,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)

        it += 1
        t += dt

        τxy_max = maximum(abs, stokes.τ.xy)
        τ_analytic = solution(εbg, t, G0, η0)
        push!(τII, τxy_max)
        push!(sol, τ_analytic)
        push!(ttot, t)

        println("it = $it; t = $t; max|τxy| = $τxy_max; analytic = $τ_analytic\n")

        write_vtk_step!(
            VTK, it, t, xvi, xci, stokes, phase_ratios, Vx_v, Vy_v, zeros_c, τ_analytic
        )
    end

    return (; ttot, τII, sol, stokes)
end

# -----------------------------------------------------------------------------
nx = 32
ny = 32
lx = 150.0e3
ly = 150.0e3
origin = (-lx / 2, -ly / 2)
V_top = 4.0e-9   # m/s (Herrendörfer V_top); bottom is no-slip (0)
dt = 500.0       # s
nsteps = 5

VTK = (;
    do_vtk = true,
    folder = joinpath(@__DIR__, "VTK"),
    vtk_dir = joinpath(@__DIR__, "VTK", "vtk"),
    pvd_name = "simple_shear_herrendorfer",
    vtk_every = 1,
)

# NOTE: do not pass `periodx` to `init_global_grid`. X-periodicity is carried by the
# velocity BCs (seam momentum row). This miniapp runs on a single rank.
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

@time main(
    igg;
    nx = nx,
    ny = ny,
    lx = lx,
    ly = ly,
    origin = origin,
    V_top = V_top,
    dt = dt,
    nsteps = nsteps,
    VTK = VTK,
)
