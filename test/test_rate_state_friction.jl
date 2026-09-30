push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D

using ParallelStencil, ParallelStencil.FiniteDifferences2D
const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC, JustPIC._2D

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

# Fill phase_ratios centers/vertices with a single RSF-enabled phase
function init_rsf_phases!(phase_ratios)
    ni = size(phase_ratios.center)
    @parallel (@idx ni) _fill_phase1!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _fill_phase1!(phase_ratios.vertex)
    return nothing
end

@parallel_indices (i, j) function _fill_phase1!(phases)
    @index phases[1, i, j] = 1.0
    @index phases[2, i, j] = 0.0
    return nothing
end

@testset "Rate-and-State Friction" begin

    @testset "RC formulas" begin
        r = RateStateFriction(0.0, 0.2, 4.0e-9, 0.011, 0.017, 0.01, 0.0, 500.0)
        Ω_old = 40.0
        P = 1.0e8
        ε = 1.0e-12
        dt = 500.0

        τ = compute_stress_frozen_Ω(r; ε = ε, Ω_old = Ω_old, P = P)
        @test τ > 0
        @test isfinite(τ)

        # frozen stress must not depend on dt
        τ2 = compute_stress_frozen_Ω(r; ε = ε, Ω_old = Ω_old, P = P, dt = 1.0e6)
        @test τ ≈ τ2

        Ω_new = update_Ω(r; ε = ε, Ω_old = Ω_old, dt = dt)
        @test isfinite(Ω_new)
        @test abs(Ω_new - Ω_old) < 50

        τ_upd = compute_stress(r; ε = ε, Ω_old = Ω_old, P = P, dt = dt)
        @test isfinite(τ_upd)

        ε_back = compute_strain_rate(r; τ = τ, Ω_old = Ω_old, P = P)
        @test isfinite(ε_back)
        @test ε_back > 0

        dt_rsf = compute_dt_ratestate(r; ε = ε, Ω_old = Ω_old, dt = dt)
        @test dt_rsf > 0
        @test isfinite(dt_rsf)
    end

    @testset "LaMEM-style a/b profiles" begin
        xk = (-47000.0, -43000.0, 33000.0, 37000.0)
        vk = (0.001, 0.017, 0.017, 0.001)
        @test get_rsf_profile(-1.0e5, xk, vk) ≈ 0.001
        @test get_rsf_profile(1.0e5, xk, vk) ≈ 0.001
        @test get_rsf_profile(-45000.0, xk, vk) ≈ 0.009 atol = 1.0e-12
        @test get_rsf_profile(0.0, xk, vk) ≈ 0.017
    end

    @testset "phase auto-enable via a_rsf" begin
        RSF = (
            enabled = true,
            loc = :center,
            V0 = 4.0e-9,
            dt_min = 1.0e-2,
            dt_max = 1.0e7,
            Phase1 = (
                a_rsf = 0.011,
                b_rsf = 0.017,
                mu0_rsf = 0.2,
                D_rs = 0.01,
                state_rsf_init = 40.0,
            ),
            Phase2 = (
                b_rsf = 0.001,  # no a_rsf → RSF off
            ),
        )
        ctrl = build_rate_state_controller(RSF; nphases = 2, di = (500.0, 500.0))
        @test ctrl.enabled
        @test ctrl.phases[1].enabled
        @test !ctrl.phases[2].enabled
        @test ctrl.phases[1].a ≈ 0.011
        @test ctrl.phases[1].Wf ≈ 500.0

        bad = (
            enabled = true,
            Phase1 = (a_rsf = 0.011, b_rsf = 0.017),
        )
        @test_throws ErrorException build_rate_state_controller(bad; nphases = 1, di = (1.0, 1.0))
    end

    @testset "APT smoke with RSF" begin
        nx, ny = 8, 8
        lx, ly = 1.0e4, 1.0e4
        ni = nx, ny
        li = lx, ly
        origin = 0.0, 0.0
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
        grid = Geometry(ni, li; origin = origin)
        (; xci) = grid
        di_min = minimum.(grid.di.vertex)

        η0 = LinearViscous(; η = 5.0e26)
        el = ConstantElasticity(; G = 3.0e10, ν = 0.25)
        rheology = (
            SetMaterialParams(;
                Name = "Media",
                Phase = 1,
                Density = ConstantDensity(; ρ = 2700.0),
                CompositeRheology = CompositeRheology((η0, el)),
                Elasticity = SetConstantElasticity(; G = 3.0e10, ν = 0.25),
                Gravity = ConstantGravity(; g = 0.0),
            ),
            SetMaterialParams(;
                Name = "Fault",
                Phase = 2,
                Density = ConstantDensity(; ρ = 2700.0),
                CompositeRheology = CompositeRheology((η0, el)),
                Elasticity = SetConstantElasticity(; G = 3.0e10, ν = 0.25),
                Gravity = ConstantGravity(; g = 0.0),
            ),
        )

        stokes = StokesArrays(backend_JR, ni)
        pt_stokes = PTStokesCoeffs(li, di_min; ϵ_abs = 1.0e-3, ϵ_rel = 1.0e-3, Re = 20.0, r = 0.7)
        stokes.P .= 5.0e6  # LaMEM p_shift; RSF needs P > 0
        stokes.viscosity.η .= 5.0e26
        stokes.viscosity.η_vep .= 5.0e26

        phase_ratios = PhaseRatios(backend, length(rheology), ni)
        init_rsf_phases!(phase_ratios)

        RSF = (
            enabled = true,
            loc = :center,
            V0 = 4.0e-9,
            dt_min = 1.0e-2,
            dt_max = 1.0e7,
            Phase1 = (
                a_rsf = 0.011,
                b_rsf = 0.017,
                mu0_rsf = 0.2,
                D_rs = 0.01,
                state_rsf_init = 40.0,
            ),
            Phase2 = (
                a_rsf = 0.011,
                b_rsf = 0.001,
                mu0_rsf = 0.2,
                D_rs = 0.01,
                state_rsf_init = -1.0,
            ),
        )
        ctrl = build_rate_state_controller(RSF; nphases = 2, di = di_min)
        fields = RateStateArrays(ni; loc = :center)
        init_rate_state_fields!(fields, ctrl, phase_ratios, xci)
        Ω_before = copy(Array(fields.Ω))
        @test all(Ω_before .≈ 40.0)

        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true),
        )
        εbg = 1.0e-14
        stokes.V.Vx .= PTArray(backend_JR)([x * εbg for x in grid.xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([-y * εbg for _ in 1:(nx + 2), y in grid.xvi[2]])
        flow_bcs!(stokes, flow_bcs)
        update_halo!(@velocity(stokes)...)

        ρg = ntuple(_ -> @zeros(ni...), Val(2))
        dt = 500.0
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

        @test all(Array(fields.Ω_old) .≈ 40.0)

        out = solve!(
            stokes,
            pt_stokes,
            grid,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                verbose = false,
                iterMax = 500.0,
                nout = 100.0,
                viscosity_cutoff = (1.0e10, 1.0e28),
                rsf = (; ctrl = ctrl, fields = fields),
            ),
        )

        @test !any(isnan, Array(stokes.τ.II))
        @test !any(isnan, Array(fields.Ω))
        @test !any(isnan, Array(stokes.viscosity.η))
        @test all(Array(fields.Ω) .== Array(fields.Ω_old))
        @test hasproperty(out, :dt_rsf)
        @test isfinite(out.dt_rsf)
        @test out.dt_rsf > 0
        @test hasproperty(out, :Vp_max)
        # miniapp policy: clamp only when selecting RSF branch
        dt_next = if out.dt_rsf < ctrl.dt_rsf_switch
            clamp(out.dt_rsf, ctrl.dt_min, ctrl.dt_max)
        else
            compute_dt(stokes, di_min, ctrl.dt_max, igg)
        end
        @test dt_next ≥ ctrl.dt_min
        @test dt_next ≤ ctrl.dt_max

        finalize_global_grid(; finalize_MPI = false)
    end
end
