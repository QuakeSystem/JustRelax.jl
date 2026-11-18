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

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

@testset "eigenstrain RSF law" begin
    G, dt, Wf = 3.0e10, 1.0e3, 500.0
    h, V = 1.0e6, 1.0e-6
    # G Δt V / Wf = 6e4 Pa of unloading at ε = 0
    @test eigenstrain_trial_stress(h, 0.0, V, G, dt, Wf) ≈ h - 6.0e4
    @test eigenstrain_trial_stress(h, 0.0, 0.0, G, dt, Wf) ≈ h
    @test eigenstrain_trial_stress(h, 0.0, 1.0, G, dt, Wf) == 0.0
    @test eigenstrain_trial_stress(0.0, 0.0, V, G, dt, Wf) == 0.0

    a, b, μ₀, V₀, σ, V_init = 0.011, 0.017, 0.2, 1.0e-6, 5.0e6, 1.0e-6
    proto = RateStateFriction(0.0, μ₀, V₀, a, b, 0.01, 0.0, Wf)
    τ0 = eigenstrain_steady_stress(a, σ, μ₀, b, V₀, V_init)
    ψ = eigenstrain_ψ_for_slip(a, b, μ₀, V₀, V_init, τ0, σ)
    @test eigenstrain_slip_rate(proto, a, b, τ0, ψ, σ) ≈ V_init rtol = 1.0e-8
    @test eigenstrain_slip_rate(proto, a, b, τ0, ψ, σ, 1.0e-8) == 1.0e-8
    @test eigenstrain_slip_rate(proto, a, b, NaN, ψ, σ) == 0.0

    # At the frozen-stress rate the slope term vanishes and the bracket stays there.
    slope = G * dt / Wf
    @test eigenstrain_consistent_rate(proto, a, b, τ0, ψ, σ, V_init, 1000.0, slope) ≈
        V_init rtol = 1.0e-6
    # A traction the raw law would turn into a huge rate must come back consistent
    # with τ - slope (V - V_lag), and must not hit the cap.
    τ_hi = 10 * τ0
    V_hi = eigenstrain_consistent_rate(proto, a, b, τ_hi, ψ, σ, V_init, 1000.0, slope)
    τ_at = τ_hi - slope * (V_hi - V_init)
    @test 0 < V_hi < 1000
    @test τ_at > 0
    @test eigenstrain_slip_rate(proto, a, b, τ_at, ψ, σ) ≈ V_hi rtol = 1.0e-4

    ψ_new = eigenstrain_age_state(proto, a, b, ψ, V_init, 1.0e5)
    r = RateStateFriction(0.0, μ₀, V₀, a, b, 0.01, 0.0, Wf)
    @test ψ_new ≈ update_Ω(r; Ω_old = ψ, dt = 1.0e5, Vp = V_init)
    @test eigenstrain_age_state(proto, a, b, ψ, 0.0, 1.0e5) ≈
        update_Ω(r; Ω_old = ψ, dt = 1.0e5, Vp = 0.0)
end

@testset "DYREL 2D inertial eigenstrain fault row" begin
    init_mpi = !JustRelax.MPI.Initialized()
    nx, ny = 4, 4
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)
    ni = (nx, ny)
    li = (2.0e3, 2.0e3)          # dy = 500 m
    Wf = 500.0
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    (; xvi) = grid

    G, η, ρ, ν = 3.0e10, 1.0e26, 2700.0, 0.25
    a, b, μ₀, V₀, L = 0.011, 0.017, 0.2, 1.0e-6, 0.01
    σ, V_init = 5.0e6, 1.0e-6
    dt = 0.2 * L / V_init
    proto = RateStateFriction(0.0, μ₀, V₀, a, b, L, 0.0, Wf)
    τ0 = eigenstrain_steady_stress(a, σ, μ₀, b, V₀, V_init)

    el = ConstantElasticity(; G = G, ν = ν)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η), el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, τ0)
    fill!(stokes.τ_o.xy, τ0)
    fill!(stokes.ε.xy, 0.0)
    fill!(stokes.V.Vx, 0.0)
    fill!(stokes.V.Vy, 0.0)
    fill!(stokes.V0.Vx, 0.0)
    fill!(stokes.V0.Vy, 0.0)
    @views stokes.V.Vx[:, end] .= V_init

    ez = eigenstrain_rsf(backend, ni; Wf = Wf, G = G, σ = σ, law = proto)
    # x <= 500 m is prescribed plate rate (mask 2): stress does not change V, and aging skips it.
    j_creep = init_eigenstrain_row!(
        ez, xvi; y_fault = 1.0e3, a = a, b = b, ψ = 0.0, V_lag = V_init, x_creep = 500.0,
    )
    @test Array(ez.mask)[1, j_creep] == 2
    @test Array(ez.mask)[3, j_creep] == 1
    set_eigenstrain_ψ_from_stress!(ez, τ0, V_init)
    @test Array(ez.ψ)[1, j_creep] == 0.0
    τ_hi = Array(stokes.τ.xy)
    τ_hi[1, j_creep] = 1.0e8
    copyto!(stokes.τ.xy, τ_hi)
    invert_rsf_eigenstrain!(ez, stokes)
    @test Array(ez.V)[1, j_creep] == V_init
    @test Array(ez.V)[3, j_creep] ≈ V_init rtol = 1.0e-6
    age_rsf_eigenstrain!(ez, 1.0e6)
    @test Array(ez.ψ)[1, j_creep] == 0.0
    fill!(stokes.τ.xy, τ0)
    fill!(stokes.τ_o.xy, τ0)

    j_fault = init_eigenstrain_row!(
        ez, xvi; y_fault = 1.0e3, a = a, b = b, ψ = 0.0, V_lag = V_init,
    )
    set_eigenstrain_ψ_from_stress!(ez, τ0, V_init)
    @test j_fault == 3
    @test count(>(0.5), Array(ez.mask)) == nx + 1

    # Direct eigenstrain: frozen slip unloads the fault row and leaves the host alone.
    apply_rsf_eigenstrain!(stokes, ez, dt)
    τ_after = Array(stokes.τ.xy)
    τ_expected = eigenstrain_trial_stress(τ0, 0.0, V_init, G, dt, Wf)
    @test all(isapprox.(τ_after[:, j_fault], τ_expected; rtol = 1.0e-10))
    @test all(isapprox.(τ_after[:, 1], τ0; rtol = 1.0e-10))
    @test τ_expected < τ0

    fill!(stokes.τ.xy, τ0)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5, CFL = 0.9)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    out = eigenstrain_physical_step!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg, ez;
        inertia = true,
        ρ_inertia = ρ,
        solve_kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 4.0e3,
            iterMax_PH = 4,
            nout = 100,
            rel_drop = 1.0e-3,
            linear_viscosity = true,
            update_material = false,
        ),
    )
    @test out.passes ≥ 1
    @test isfinite(out.rel)
    @test out.V_max ≥ 0
    @test all(isfinite, Array(stokes.τ.xy))
    @test all(isfinite, Array(ez.V))
    @test all(>=(0), Array(ez.V))
    # Inertia history is the velocity of this accepted step, not the previous pass.
    @test Array(stokes.V0.Vx) ≈ Array(stokes.V.Vx)
    @test Array(stokes.V0.Vy) ≈ Array(stokes.V.Vy)
    # Slip rate was committed onto the lag for the next step.
    @test Array(ez.V_lag) ≈ Array(ez.V)
    @test maximum(Array(ez.V)[:, j_fault]) > 0
end
