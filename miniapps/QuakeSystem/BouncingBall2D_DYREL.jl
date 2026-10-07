#=
LaMEM-style bouncing ball + elastic floor block (DYREL + physical inertia).

Geometry (SI, vertical = y), matching `bouncing_ball_block.dat`:
  - Domain: x ∈ [-5e3, 5e3] m, y ∈ [-10e3, 0] m
  - Soft matrix (phase 1)
  - Dense elastic ball (phase 2) centred at (0, -4e3), radius 2e3 m
  - Elastic floor block (phase 3) for y ≤ -6.5e3 m

Eulerian phases (no marker advection yet) — the continuum deforms in place;
use ParaView to watch Vy / phase / density under inertia.

Default: 64×64, VTK → `QuakeSystem/VTK_bouncing_ball/`
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

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

# Phase layout: 1 = matrix, 2 = ball, 3 = floor
@parallel_indices (i, j) function _init_bouncing_phases!(phases, coords, xc_ball, yc_ball, r_ball, y_floor)
    x = coords[1][i]
    y = coords[2][j]
    if y ≤ y_floor
        @index phases[1, i, j] = 0.0
        @index phases[2, i, j] = 0.0
        @index phases[3, i, j] = 1.0
    elseif (x - xc_ball)^2 + (y - yc_ball)^2 ≤ r_ball^2
        @index phases[1, i, j] = 0.0
        @index phases[2, i, j] = 1.0
        @index phases[3, i, j] = 0.0
    else
        @index phases[1, i, j] = 1.0
        @index phases[2, i, j] = 0.0
        @index phases[3, i, j] = 0.0
    end
    return nothing
end

@parallel_indices (i, j) function _fill_density!(ρ, phases, ρ_mat, ρ_ball, ρ_floor)
    ρ[i, j] =
        @index(phases[1, i, j]) * ρ_mat +
        @index(phases[2, i, j]) * ρ_ball +
        @index(phases[3, i, j]) * ρ_floor
    return nothing
end

phase_map(phase_ratios_loc) = [argmax(p) for p in Array(phase_ratios_loc)]

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

function write_vtk_step!(VTK, it, t, xvi, xci, stokes, phase_ratios, ρ_c, Vx_v, Vy_v)
    VTK.do_vtk || return nothing
    vtk_every = Int(get(VTK, :vtk_every, 1))
    (it == 1 || rem(it, vtk_every) == 0) || return nothing

    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    data_v = (;
        Vx = Array(Vx_v),
        Vy = Array(Vy_v),
        phase = phase_map(phase_ratios.vertex),
    )
    data_c = (;
        P = Array(stokes.P),
        η = Array(stokes.viscosity.η),
        τII = Array(stokes.τ.II),
        εII = Array(stokes.ε.II),
        ρ = Array(ρ_c),
        phase = phase_map(phase_ratios.center),
    )
    velocity_v = (Array(Vx_v), Array(Vy_v))
    path_vtk = joinpath(VTK.vtk_dir, "vtk_" * lpad("$it", 6, "0"))
    save_vtk(
        path_vtk, xvi, xci, data_v, data_c, velocity_v;
        t = t, pvd = joinpath(VTK.vtk_dir, VTK.pvd_name),
    )
    println("  VTK → $(joinpath(VTK.vtk_dir, VTK.pvd_name)).pvd  (it=$it, t=$t)")
    return nothing
end

function mean_ball_Vy(stokes, f_ball)
    Vy = Array(stokes.V.Vy)
    nx, ny = size(stokes.P)
    s, n = 0.0, 0
    for j in 1:ny, i in 1:nx
        if f_ball[i, j] ≥ 0.5
            s += Vy[i + 1, j + 1]
            n += 1
        end
    end
    return n == 0 ? 0.0 : s / n
end

function main(
        igg;
        nx = 64,
        ny = 64,
        dt = 5.0e-2,
        nsteps = 20,
        VTK = nothing,
    )
    # ---- LaMEM bouncing_ball_block geometry ----
    lx, ly = 10.0e3, 10.0e3
    origin = (-5.0e3, -10.0e3)
    ni = (nx, ny)
    li = (lx, ly)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid

    g = 10.0
    ρ_mat, ρ_ball, ρ_floor = 1.0, 700.0, 3000.0
    # Softened vs LaMEM η_mat=1e-5 so DYREL stays stable; ball/floor nearly elastic
    η_mat = 1.0e2
    η_el = 1.0e12
    G_ball, G_floor = 3.0e10, 2.0e10
    ν = 0.3
    xc_ball, yc_ball, r_ball = 0.0, -4.0e3, 2.0e3
    y_floor = -6.5e3

    el_ball = ConstantElasticity(; G = G_ball, ν = ν)
    el_floor = ConstantElasticity(; G = G_floor, ν = ν)
    rheology = (
        SetMaterialParams(;
            Name = "Matrix",
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ_mat),
            Gravity = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_mat),)),
        ),
        SetMaterialParams(;
            Name = "Ball",
            Phase = 2,
            Density = ConstantDensity(; ρ = ρ_ball),
            Gravity = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_el), el_ball)),
            Elasticity = el_ball,
        ),
        SetMaterialParams(;
            Name = "Floor",
            Phase = 3,
            Density = ConstantDensity(; ρ = ρ_floor),
            Gravity = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_el), el_floor)),
            Elasticity = el_floor,
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, 3, ni)
    @parallel (@idx ni) _init_bouncing_phases!(
        phase_ratios.center, xci, xc_ball, yc_ball, r_ball, y_floor
    )
    @parallel (@idx ni .+ 1) _init_bouncing_phases!(
        phase_ratios.vertex, xvi, xc_ball, yc_ball, r_ball, y_floor
    )

    ρ_inertia = @zeros(ni...)
    @parallel (@idx ni) _fill_density!(ρ_inertia, phase_ratios.center, ρ_mat, ρ_ball, ρ_floor)

    phases_c = Array(phase_ratios.center)
    f_ball = [phases_c[i, j][2] for i in 1:nx, j in 1:ny]

    isnothing(VTK) && (VTK = (;
            do_vtk = true,
            folder = joinpath(@__DIR__, "VTK_bouncing_ball"),
            vtk_dir = joinpath(@__DIR__, "VTK_bouncing_ball", "vtk"),
            pvd_name = "bouncing_ball",
            vtk_every = 2,
        ))
    prepare_vtk!(VTK)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = false, right = false, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    # Rough lithostatic seed P ≈ ρ g · depth (stabilizes first steps)
    y_top = origin[2] + ly
    ρ_cpu = Array(ρ_inertia)
    P_cpu = [ρ_cpu[i, j] * g * (y_top - xci[2][j]) for i in 1:nx, j in 1:ny]
    copyto!(stokes.P, P_cpu)
    fill!(stokes.V.Vx, 0.0)
    fill!(stokes.V.Vy, 0.0)
    fill!(stokes.V0.Vx, 0.0)
    fill!(stokes.V0.Vy, 0.0)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    viscosity_cutoff = (1.0e-2, 1.0e14)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    compute_ρg!(ρg[end], phase_ratios, rheology, args)

    dyrel = DYREL(
        backend, stokes, rheology, phase_ratios, grid.di, dt;
        ϵ = 1.0e-5, CFL = 0.5,
    )

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    println("Bouncing ball  ni=$ni  dt=$dt  nsteps=$nsteps  inertia=ON")
    println("  ball @ ($xc_ball, $yc_ball) r=$r_ball; floor y≤$y_floor")

    t, it = 0.0, 0
    for _ in 1:nsteps
        out = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 5.0e3,
                total_iterMax = 8.0e3,
                iterMax_PH = 30,
                nout = 100,
                rel_drop = 1.0e-2,
                linear_viscosity = false,
                update_material = true,
                viscosity_cutoff = viscosity_cutoff,
                inertia = true,
                ρ_inertia = ρ_inertia,
            ),
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)

        it += 1
        t += dt
        Vy_b = mean_ball_Vy(stokes, f_ball)
        println(
            "it=$it  t=$(round(t; digits=3))  " *
                "Vy_ball=$(round(Vy_b; sigdigits=4))  " *
                "iter=$(out.iter)  " *
                "err=$(isempty(out.err_evo_tot) ? NaN : round(out.err_evo_tot[end]; sigdigits=3))"
        )
        write_vtk_step!(VTK, it, t, xvi, xci, stokes, phase_ratios, ρ_inertia, Vx_v, Vy_v)
    end

    println("Done. Open $(joinpath(VTK.vtk_dir, VTK.pvd_name)).pvd in ParaView.")
    return nothing
end

# -----------------------------------------------------------------------------
nx = ny = 64
dt = 5.0e-2
nsteps = 20
VTK = (;
    do_vtk = true,
    folder = joinpath(@__DIR__, "VTK_bouncing_ball"),
    vtk_dir = joinpath(@__DIR__, "VTK_bouncing_ball", "vtk"),
    pvd_name = "bouncing_ball",
    vtk_every = 1,
)

igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

@time main(igg; nx = nx, ny = ny, dt = dt, nsteps = nsteps, VTK = VTK)
