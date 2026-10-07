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

import JustRelax.JustRelax2D as JR2K

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

# Fast continuum column check for LaMEM-style inertia (not a full marker bounce).
# Caps DYREL iterations tightly so a bad step fails fast instead of grinding.
@testset "DYREL 2D bouncing ball (inertia)" begin
    init_mpi = !JustRelax.MPI.Initialized()
    nx, ny = 8, 16
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)
    ni = (nx, ny)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    (; _di) = grid

    @testset "residual carries ρ(V−V0)/dt" begin
        stokes = StokesArrays(backend, ni)
        fill!(stokes.V.Vx, 0.0)
        fill!(stokes.V.Vy, 0.0)
        fill!(stokes.V0.Vx, 0.0)
        fill!(stokes.V0.Vy, 0.0)
        @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 1.0
        ρ_inertia = @ones(ni...)
        ρg = @zeros(ni...), @zeros(ni...)
        _inv_dt = 10.0
        @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry,
            stokes.V.Vx, stokes.V.Vy,
            stokes.V0.Vx, stokes.V0.Vy,
            stokes.P, stokes.ΔPψ,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy,
            ρg...,
            ρ_inertia, _di.center, _di.vertex, 0.0, _inv_dt,
        )
        Ry = Array(stokes.R.Ry)
        @test all(isapprox.(Ry, -10.0; atol = 1.0e-12))
    end

    @testset "column: V0 snapshot + finite residual" begin
        ρ, η, dt = 1.0, 1.0, 0.05
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = ρ),
                Gravity = ConstantGravity(; g = 0.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = η),)),
            ),
        )
        phase_ratios = PhaseRatios(backend_JP, 1, ni)
        @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
        @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = false),
            no_slip = (left = false, right = false, top = false, bot = true),
            periodic = (left = false, right = false, top = false, bot = false),
        )
        stokes = StokesArrays(backend, ni, flow_bcs)
        fill!(stokes.V.Vx, 0.0)
        fill!(stokes.V.Vy, -0.5)
        fill!(stokes.V0.Vx, 0.0)
        fill!(stokes.V0.Vy, -0.5)
        flow_bcs!(stokes, flow_bcs)
        update_halo!(@velocity(stokes)...)

        ρg = @zeros(ni...), @zeros(ni...)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
        # Mild outer tolerance + hard iter caps: column should finish in ≪1 s of PT work
        dyrel = DYREL(
            backend, stokes, rheology, phase_ratios, grid.di, dt;
            ϵ = 1.0e-4, CFL = 0.5,
        )

        out = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 1.0e3,
                total_iterMax = 1.0e3,
                iterMax_PH = 10,
                nout = 100,
                rel_drop = 1.0e-2,
                linear_viscosity = true,
                update_material = false,
                inertia = true,
                ρ_inertia = ρ,
            ),
        )

        @test isempty(out.err_evo_tot) || (isfinite(out.err_evo_tot[end]) && out.err_evo_tot[end] < 1.0e2)
        @test Array(stokes.V0.Vx) ≈ Array(stokes.V.Vx)
        @test Array(stokes.V0.Vy) ≈ Array(stokes.V.Vy)
        @test all(isfinite, Array(stokes.V.Vy))
        @test out.iter ≤ 1_200   # total_iterMax=1e3 plus one PH overshoot
    end
end
