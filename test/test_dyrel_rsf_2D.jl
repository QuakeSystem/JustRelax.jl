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

@parallel_indices (i, j) function _fill_fault_phases!(phases, j_lo, j_hi)
    if j_lo ≤ j ≤ j_hi
        @index phases[1, i, j] = 0.0
        @index phases[2, i, j] = 1.0
    else
        @index phases[1, i, j] = 1.0
        @index phases[2, i, j] = 0.0
    end
    return nothing
end

@testset "RSF constitutive τ → Vp (LaMEM)" begin
    # Media-like (locked Ω): tiny stress → negligible slip rate
    r_media = RateStateFriction(0.0, 0.2, 4.0e-9, 0.011, 0.017, 0.01, 0.0, 500.0)
    P = 5.0e6
    Vp_low = compute_Vp_from_stress(r_media; τ = 0.4, Ω = 40.0, P = P)
    @test Vp_low < 1.0e-30
    η_lock, Vp_lock = rsf_viscosity_from_stress(r_media; τ = 0.4, Ω_old = 40.0, P = P)
    @test isinf(η_lock) || η_lock > 1.0e40
    @test Vp_lock == Vp_low

    # Fault-like (Ω = -1): at τ ~ 1 MPa, Vp ~ V₀
    r_fault = RateStateFriction(0.0, 0.2, 4.0e-9, 0.011, 0.001, 0.01, 0.0, 500.0)
    Vp_act = compute_Vp_from_stress(r_fault; τ = 1.0e6, Ω = -1.0, P = P)
    @test isapprox(Vp_act, 4.0e-9; rtol = 0.2)
    η_act, _ = rsf_viscosity_from_stress(r_fault; τ = 1.0e6, Ω_old = -1.0, P = P)
    @test isfinite(η_act) && η_act > 0
    # η_rsf = τ Wf / Vp; DIIrsf = Vp/(2 Wf) recovers τ = 2 η_rsf DIIrsf
    @test isapprox(η_act, 1.0e6 * 500.0 / Vp_act; rtol = 1.0e-12)

    # Asinh(ε) path overestimates activity while locked (documents why DYREL uses stress path)
    ε = 4.0e-9 / 150.0e3
    _, η_asinh, Vp_ε = rsf_frozen_stress_viscosity(r_media; ε = ε, Ω_old = 40.0, P = P)
    @test Vp_ε ≈ 2 * 500.0 * ε
    @test isfinite(η_asinh) && η_asinh < 1.0e21
end

@testset "RSF LaMEM η bisection" begin
    r = RateStateFriction(0.0, 0.2, 4.0e-9, 0.011, 0.001, 0.01, 0.0, 500.0)
    P = 5.0e6
    Gdt = 3.0e10 * 500.0
    η_creep = 1.0e28
    DII = 4.0e-9 / 150.0e3

    # Locked media: η_eff ≈ Maxwell η_ve, residual ~ 0, creep buffer unchanged
    r_lock = RateStateFriction(0.0, 0.2, 4.0e-9, 0.011, 0.017, 0.01, 0.0, 500.0)
    η_eff, η_out, Vp, τ = rsf_effective_viscosity(
        r_lock; DII = DII, η_creep = η_creep, Gdt = Gdt, Ω_old = 40.0, P = P
    )
    η_ve = 1 / (1 / η_creep + 1 / Gdt)
    @test isapprox(η_eff, η_ve; rtol = 1.0e-6)
    @test η_out ≈ η_creep
    @test Vp < 1.0e-20
    @test abs(rsf_cons_eq_residual(η_eff, DII, η_creep, Gdt, r_lock, 40.0, P)) ≤ 1.0e-12 * DII

    # Active fault Ω=-1 at larger DII / stress: residual closed, η_eff < η_ve
    DII_act = 1.0e-10
    η_eff_a, η_out_a, Vp_a, τ_a = rsf_effective_viscosity(
        r; DII = DII_act, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P
    )
    @test isfinite(η_eff_a) && η_eff_a > 0
    @test η_eff_a < η_ve
    @test abs(rsf_cons_eq_residual(η_eff_a, DII_act, η_creep, Gdt, r, -1.0, P)) ≤ 1.0e-8 * DII_act
    # Creep buffer maps back to η_ve ≈ η_eff
    η_ve_out = 1 / (1 / η_out_a + 1 / Gdt)
    @test isapprox(η_ve_out, η_eff_a; rtol = 1.0e-6)
    @test τ_a ≈ 2 * η_eff_a * DII_act
    @test Vp_a ≈ compute_Vp_from_stress(r; τ = τ_a, Ω = -1.0, P = P)
end

@testset "DYREL 2D RSF" begin
    init_mpi = !JustRelax.MPI.Initialized()
    nx, ny = 16, 16
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    ni = (nx, ny)
    li = (1.0e3, 1.0e3)
    origin = (0.0, 0.0)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid
    dt = 500.0
    V_top = 4.0e-9
    P0 = 5.0e6

    η0 = 1.0e23
    G0 = 3.0e10
    el = ConstantElasticity(; G = G0, ν = 0.25)
    visc = LinearViscous(; η = η0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    # horizontal fault band through mid-domain
    j_lo, j_hi = ny ÷ 2 - 1, ny ÷ 2 + 1
    @parallel (@idx ni) _fill_fault_phases!(phase_ratios.center, j_lo, j_hi)
    @parallel (@idx ni .+ 1) _fill_fault_phases!(phase_ratios.vertex, j_lo, j_hi)

    RSF = (
        enabled = true,
        loc = :both,
        V0 = V_top,
        dt_min = 1.0e-2,
        dt_max = 1.0e7,
        dt_rsf_switch = 1.0e9,
        G = G0,
        ν = 0.25,
        Phase2 = (
            a_rsf = 0.011,
            b_rsf = 0.017,
            mu0_rsf = 0.2,
            D_rs = 0.01,
            Wf = 50.0,
            # Moderate Ω so healing V₀ dt / L is visible in Float64 (Ω=40 is sticky)
            state_rsf_init = 1.0,
            λ = 0.0,
            C = 0.0,
        ),
    )
    ctrl = build_rate_state_controller(RSF; nphases = 2, di = (li[1] / nx, li[2] / ny))
    @test ctrl.enabled
    @test ctrl.phases[1].enabled == false
    @test ctrl.phases[2].enabled == true
    @test ctrl.loc === :both

    rsf_fields = RateStateArrays(backend, ni; loc = :both)
    init_rate_state_fields!(rsf_fields, ctrl, phase_ratios, xci, xvi)
    @test maximum(Array(rsf_fields.rsf_mask)) ≈ 1.0
    @test maximum(Array(rsf_fields.rsf_mask_v)) ≈ 1.0
    @test size(rsf_fields.Ωv) == ni .+ 1

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, P0)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    compute_viscosity!(stokes, phase_ratios, args, rheology, (1.0e10, 1.0e28))
    yVx = grid.xi_vel[1][2]
    ybot, ytop = origin[2], origin[2] + li[2]
    stokes.V.Vx .= PTArray(backend)([
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi[1], y in yVx
        ])
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5)
    rsf_bundle = (; ctrl = ctrl, fields = rsf_fields)

    # Ω_old must stay frozen through the solve; only update after convergence
    Ω_before = copy(Array(rsf_fields.Ω_old))
    Ωv_before = copy(Array(rsf_fields.Ω_old_v))
    out = solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 20.0e3,
            nout = 50,
            rel_drop = 1.0e-2,
            linear_viscosity = true,
            viscosity_cutoff = (1.0e10, 1.0e28),
            rsf = rsf_bundle,
        )
    )

    @test hasproperty(out, :dt_rsf)
    @test isfinite(out.dt_rsf)
    @test out.dt_rsf > 0
    # healing advances Ω when Ω is O(1); locked Ω≃40 would not move in Float64
    Ω_after = Array(rsf_fields.Ω_old)
    Ωv_after = Array(rsf_fields.Ω_old_v)
    @test any(abs.(Ω_after .- Ω_before) .> 0) || any(abs.(Ωv_after .- Ωv_before) .> 0)
    # asinh envelope stored for diagnostics on both locations
    @test maximum(Array(rsf_fields.τ_rsf)) > 0 || maximum(Array(rsf_fields.τ_rsf_v)) > 0
    # Early Couette loading: τ elastic, stress-based Vp ≪ V₀, η not collapsed by asinh fold
    εbg = V_top / li[2]
    τ_elastic = εbg * G0 * dt  # Maxwell ≈ G ε̇ t for t ≪ η/G
    @test maximum(abs, Array(stokes.τ.xy)) < 10 * τ_elastic
    @test maximum(abs, Array(stokes.τ.xy)) < 1.0e-3 * P0 * 0.2
    @test isapprox(maximum(abs, Array(stokes.τ.xy)), τ_elastic; rtol = 0.15)
    @test out.Vp_max < 1.0e-3 * V_top
    @test minimum(Array(stokes.viscosity.η)) > 1.0e22
    @test minimum(Array(stokes.viscosity.ηv)) > 1.0e22

    finalize_global_grid(; finalize_MPI = false)
end
