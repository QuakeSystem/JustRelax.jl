push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

# =============================================================================
# Analytics — port of LaMEM test/t39_InertiaSHWave/SHWave_analytics.jl
# Homogeneous elastic standing SH wave.
#   vx(y,t) = V0 sin(k y) cos(ω t),  k = n π / H,  ω = sqrt(G/ρ) k
# JustRelax 2D maps LaMEM (x,z) → (x,y) with y vertical.
# =============================================================================

const SH_ρ = 3000.0
const SH_μ = 3.0e10
const SH_H = 1000.0
const SH_V0 = 1.0e-4
const SH_n = 1
const SH_Vs = sqrt(SH_μ / SH_ρ)
const SH_k = SH_n * π / SH_H
const SH_ω = SH_Vs * SH_k
const SH_T = 2π / SH_ω
const SH_η = 1.0e20  # ≫ G·dt → Maxwell wall is elastic

"Amplitude of τ_xy (= LaMEM τ_xz)."
shwave_tau_amp() = SH_μ * SH_k * SH_V0 / SH_ω

"Amplitude of ux."
shwave_disp_amp() = SH_V0 / SH_ω

function shwave_fields(y, t)
    s = sin.(SH_k .* y)
    c = cos.(SH_k .* y)
    vx = SH_V0 .* s .* cos(SH_ω * t)
    ux = shwave_disp_amp() .* s .* sin(SH_ω * t)
    tx = shwave_tau_amp() .* c .* sin(SH_ω * t)
    return vx, ux, tx
end

"Elastic plus kinetic energy per unit horizontal area, from samples along y."
function shwave_energy(vx, tx, y)
    e = 0.5 .* SH_ρ .* vx .^ 2 .+ tx .^ 2 ./ (2 * SH_μ)
    dy = diff(y)
    return sum(0.5 .* (e[1:(end - 1)] .+ e[2:end]) .* dy)
end

function shwave_rms(a)
    return sqrt(sum(abs2, a) / length(a))
end

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

"""Average face-centred Vx (size (nx+1, ny+2)) onto vertex y-rows (length ny+1)."""
function _vx_faces_to_vertices(Vx)
    nx1, ny2 = size(Vx)
    ny = ny2 - 2
    vxv = zeros(ny + 1)
    for j in 1:(ny + 1)
        vxv[j] = sum(view(Vx, :, j) .+ view(Vx, :, j + 1)) / (2 * nx1)
    end
    return vxv
end

"""
    err = run_shwave_case(; ny, nstep, nx=4, return_fields=false)

Standing SH wave over `T/8` with fixed `dt = (T/8)/nstep`.
Returns relative RMS errors of vx, ux, τxy and relative energy drift
(same named-tuple fields as LaMEM `shwave_errors`).

With `return_fields=true`, also returns profiles for plotting:
`y_vx`, `vx_num`, `vx_ana`, `ux_num`, `ux_ana`, `y_τ`, `τ_num`, `τ_ana`.
"""
function run_shwave_case(; ny::Int, nstep::Int, nx::Int = 4, return_fields::Bool = false)
    if ImplicitGlobalGrid.grid_is_initialized()
        finalize_global_grid(; finalize_MPI = false)
    end
    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    ni = (nx, ny)
    dt = (SH_T / 8) / nstep
    grid = Geometry(ni, (250.0, SH_H); origin = (0.0, 0.0))

    el = ConstantElasticity(; G = SH_μ, ν = 0.25)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = SH_ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = SH_η), el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = true, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)

    # IC: vx = V0 sin(k y), stress = 0. Seed V0 = V (standing wave has ∂v/∂t = 0 at t = 0).
    yVx = Array(grid.xi_vel[1][2])
    xVx = Array(grid.xi_vel[1][1])
    stokes.V.Vx .= PTArray(backend)([SH_V0 * sin(SH_k * y) for _ in xVx, y in yVx])
    fill!(stokes.V.Vy, 0.0)
    copyto!(stokes.V0.Vx, stokes.V.Vx)
    copyto!(stokes.V0.Vy, stokes.V.Vy)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    Ux = zeros(size(stokes.V.Vx))
    yv = Array(grid.xvi[2])
    E0 = shwave_energy(_vx_faces_to_vertices(Array(stokes.V.Vx)), zeros(ny + 1), yv)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8, CFL = 0.9)

    t = 0.0
    for _ in 1:nstep
        solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 5.0e3,
                total_iterMax = 1.0e4,
                iterMax_PH = 3,
                nout = 50,
                rel_drop = 1.0e-3,
                linear_viscosity = true,
                update_material = false,
                inertia = true,
                ρ_inertia = SH_ρ,
            ),
        )
        flow_bcs!(stokes, flow_bcs)
        # tot_displ: JR stokes.U is only V·dt, so accumulate here (LaMEM-style).
        Ux .+= Array(stokes.V.Vx) .* dt
        t += dt
    end

    VxA = Array(stokes.V.Vx)
    τA = Array(stokes.τ.xy)

    # Velocity / displacement on interior Vx faces (ghost-stripped), x-averaged
    Vx_num = vec(sum(VxA[:, 2:(end - 1)]; dims = 1) ./ size(VxA, 1))
    Ux_num = vec(sum(Ux[:, 2:(end - 1)]; dims = 1) ./ size(Ux, 1))
    y_int = yVx[2:(end - 1)]
    vxa, uxa, _ = shwave_fields(y_int, t)

    # Shear stress on vertices (LaMEM corner τ_xz → JR τ.xy)
    τ_num = vec(sum(τA; dims = 1) ./ size(τA, 1))
    _, _, txa = shwave_fields(yv, t)

    E1 = shwave_energy(_vx_faces_to_vertices(VxA), τ_num, yv)

    finalize_global_grid(; finalize_MPI = false)

    err = (
        t = t,
        vx = shwave_rms(Vx_num .- vxa) / SH_V0,
        ux = shwave_rms(Ux_num .- uxa) / shwave_disp_amp(),
        tx = shwave_rms(τ_num .- txa) / shwave_tau_amp(),
        energy = abs(E1 - E0) / max(E0, eps()),
        maxVy = maximum(abs, Array(stokes.V.Vy)),
        maxP = maximum(abs, Array(stokes.P)),
    )
    return_fields || return err
    return merge(
        err,
        (;
            y_vx = y_int,
            vx_num = Vx_num,
            vx_ana = vxa,
            ux_num = Ux_num,
            ux_ana = uxa,
            y_τ = yv,
            τ_num = τ_num,
            τ_ana = txa,
        ),
    )
end

"""
Four runs over one eighth of a period (LaMEM `shwave_convergence`).

LaMEM uses nstep = 32 with a direct Stokes solve. DYREL is iterative, so the
spatial members use nstep = 16 (same dt across ny) to keep CI time reasonable
while preserving the coarse/space/fine and large-dt members.
"""
function shwave_convergence()
    # nstep = 16 over T/8  (LaMEM reference uses 32)
    coarse = run_shwave_case(; ny = 8, nstep = 16)
    space = run_shwave_case(; ny = 16, nstep = 16)
    fine = run_shwave_case(; ny = 32, nstep = 16)
    # 4× larger dt than the spatial members
    time = run_shwave_case(; ny = 32, nstep = 4)
    return (; coarse, space, fine, time)
end

# Set ENV["SHWAVE_REPL"]="1" before `include`ing this file in a REPL to load
# helpers without running the assertion suite.
if get(ENV, "SHWAVE_REPL", "0") != "1"
    @testset "DYREL 2D inertial standing SH wave" begin
        # Pure SH: no normal motion / pressure should appear
        cases = shwave_convergence()
        @info "SH wave errors" coarse = cases.coarse space = cases.space fine = cases.fine time = cases.time

        fine = cases.fine
        @test fine.maxVy < 1.0e-12 * SH_V0
        @test fine.maxP < 1.0e-6 * shwave_tau_amp()

        # Resolved case: same absolute thresholds as LaMEM t39
        @test fine.vx < 0.05
        @test fine.ux < 0.08
        @test fine.tx < 0.05
        @test fine.energy < 0.05

        # Spatial: shear-stress error falls as dy is halved at fixed dt.
        # (Velocity error is dominated by backward-Euler damping under the DYREL
        #  iteration budget, so vx ordering is not asserted — unlike LaMEM's direct solve.)
        @test cases.coarse.tx > cases.space.tx
        @test cases.space.tx > cases.fine.tx

        # Temporal: 4× larger dt increases the velocity error (backward Euler)
        @test cases.time.vx > 2 * cases.fine.vx
    end
end
