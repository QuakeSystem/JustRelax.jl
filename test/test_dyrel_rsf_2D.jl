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

    # Locked: dVp/dτ vanishes, so the tangent is the Maxwell viscosity.
    η_tan_lock = rsf_tangent_viscosity(
        r_lock; τ = τ, Ω = 40.0, P = P, η_creep = η_creep, Gdt = Gdt,
    )
    @test isapprox(η_tan_lock, η_ve; rtol = 1.0e-6)
    @test dVp_dτ(r_lock; τ = 0.0, Ω = 40.0, P = P) == 0

    # Loaded: slope matches Vp coth(ξ) / (a P), and η_tan is softer than the secant.
    aP = 0.011 * P
    ξ = τ_a / aP
    @test dVp_dτ(r; τ = τ_a, Ω = -1.0, P = P) ≈ Vp_a * coth(ξ) / aP rtol = 1.0e-8
    η_tan_a = rsf_tangent_viscosity(
        r; τ = τ_a, Ω = -1.0, P = P, η_creep = η_creep, Gdt = Gdt,
    )
    @test isfinite(η_tan_a) && η_tan_a > 0
    @test η_tan_a < η_eff_a

    # Relative floor η_eff ≥ Gdt/η_contrast. Default Inf reproduces the unbounded solve.
    η_eff_inf, η_out_inf, _, _ = rsf_effective_viscosity(
        r; DII = DII_act, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_contrast = Inf
    )
    @test η_eff_inf == η_eff_a
    @test η_out_inf == η_out_a
    # Locked: η_eff ≈ Gdt is above the floor, so nothing changes.
    η_eff_lock_c, η_out_lock_c, _, _ = rsf_effective_viscosity(
        r_lock; DII = DII, η_creep = η_creep, Gdt = Gdt, Ω_old = 40.0, P = P, η_contrast = 1.0e4
    )
    @test η_eff_lock_c == η_eff
    @test η_out_lock_c ≈ η_creep
    # Fast-slipping node well below the floor: pinned to Gdt/η_contrast, and the creep
    # buffer maps back to that value through the Maxwell combine.
    # At this rate the Maxwell trial stress (30 MPa) gives Vp ~ 1e220 and a stress-based
    # η_rsf below ε_floor. That guess used to be dropped as "locked", collapsing the
    # bracket onto η_ve; the solve must return the finite slipping root instead.
    DII_soft = 1.0e-6
    η_guess, _ = rsf_viscosity_from_stress(r; τ = 2 * η_ve * DII_soft, Ω_old = -1.0, P = P)
    @test η_guess < 1.0e-30
    η_eff_soft, η_out_soft, Vp_soft, τ_soft = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P
    )
    @test isfinite(η_eff_soft) && η_eff_soft > 0
    @test abs(rsf_cons_eq_residual(η_eff_soft, DII_soft, η_creep, Gdt, r, -1.0, P)) ≤ 1.0e-6 * DII_soft
    @test isfinite(Vp_soft)
    @test Vp_soft ≈ 2 * 500.0 * (DII_soft - τ_soft / (2 * Gdt)) rtol = 1.0e-6
    @test isapprox(1 / (1 / η_out_soft + 1 / Gdt), η_eff_soft; rtol = 1.0e-6)
    contrast = 2.0
    @test η_eff_soft < Gdt / contrast / 5
    η_eff_c, η_out_c, Vp_c, τ_c = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_contrast = contrast
    )
    @test η_eff_c ≈ Gdt / contrast rtol = 1.0e-12
    @test isapprox(1 / (1 / η_out_c + 1 / Gdt), η_eff_c; rtol = 1.0e-6)
    @test τ_c ≈ 2 * η_eff_c * DII_soft
    @test Vp_c ≈ compute_Vp_from_stress(r; τ = τ_c, Ω = -1.0, P = P)
    # A floor below the unbounded answer is inactive.
    η_eff_loose, _, _, _ = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_contrast = 1.0e6
    )
    @test η_eff_loose == η_eff_soft
    # The floor can never exceed the Maxwell trial viscosity.
    η_eff_tight, _, _, _ = rsf_effective_viscosity(
        r; DII = DII_act, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_contrast = 1.0 + 1.0e-12
    )
    @test η_eff_tight ≤ η_ve * (1 + 1.0e-12)

    # Log-space relaxation toward η_prev. relax = 1 or no history: unbounded result.
    η_prev = Gdt
    η_eff_r1, _, _, _ = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_prev = η_prev, relax = 1
    )
    @test η_eff_r1 == η_eff_soft
    η_eff_r0, _, _, _ = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_prev = 0.0, relax = 0.5
    )
    @test η_eff_r0 == η_eff_soft
    # relax = 0.5 lands at the geometric mean, with τ and Vp evaluated there.
    η_eff_half, η_out_half, Vp_half, τ_half = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_prev = η_prev, relax = 0.5
    )
    @test η_eff_half ≈ sqrt(η_eff_soft * η_prev) rtol = 1.0e-10
    @test η_eff_soft < η_eff_half < η_prev
    @test τ_half ≈ 2 * η_eff_half * DII_soft
    @test Vp_half ≈ compute_Vp_from_stress(r; τ = τ_half, Ω = -1.0, P = P)
    @test isapprox(1 / (1 / η_out_half + 1 / Gdt), η_eff_half; rtol = 1.0e-6)
    # Repeated relaxed folds from the same strain rate converge to the unrelaxed answer.
    η_iter = η_prev
    for _ in 1:60
        η_iter, _, _, _ = rsf_effective_viscosity(
            r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_prev = η_iter, relax = 0.5
        )
    end
    @test η_iter ≈ η_eff_soft rtol = 1.0e-8
    # A previous iterate above the Maxwell trial is clipped to it first.
    η_eff_clip, _, _, _ = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_prev = 1.0e3 * Gdt, relax = 0.5
    )
    @test η_eff_clip ≈ sqrt(η_eff_soft * η_ve) rtol = 1.0e-10

    # Absolute creep-buffer cutoff: η_eff is raised to the Maxwell image of η_min, and
    # τ / Vp are recomputed there so the Stokes stress update matches the fold.
    η_min_abs = 1.0e12
    η_ve_floor = 1 / (1 / η_min_abs + 1 / Gdt)
    @test η_eff_soft < η_ve_floor
    η_eff_abs, η_out_abs, Vp_abs, τ_abs = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_min = η_min_abs
    )
    @test η_eff_abs ≈ η_ve_floor rtol = 1.0e-12
    @test η_out_abs ≈ η_min_abs rtol = 1.0e-6
    @test τ_abs ≈ 2 * η_eff_abs * DII_soft
    @test Vp_abs ≈ compute_Vp_from_stress(r; τ = τ_abs, Ω = -1.0, P = P)
    # A floor below the unbounded answer is inactive; τ/Vp match the unclamped solve.
    η_eff_abs_lo, η_out_abs_lo, Vp_abs_lo, τ_abs_lo = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P, η_min = 1.0e6
    )
    @test η_eff_abs_lo == η_eff_soft
    @test η_out_abs_lo == η_out_soft
    @test τ_abs_lo == τ_soft
    @test Vp_abs_lo == Vp_soft
    # Absolute floor wins over a softer relative contrast floor.
    η_eff_both, _, _, _ = rsf_effective_viscosity(
        r; DII = DII_soft, η_creep = η_creep, Gdt = Gdt, Ω_old = -1.0, P = P,
        η_contrast = 1.0e6, η_min = η_min_abs,
    )
    @test η_eff_both ≈ η_ve_floor rtol = 1.0e-12
    # Preconditioner slope cannot be stiffer than the clamped secant.
    η_tan_unbounded = rsf_tangent_viscosity(
        r; τ = τ_abs, Ω = -1.0, P = P, η_creep = η_creep, Gdt = Gdt,
    )
    @test rsf_preconditioner_tangent(η_tan_unbounded, η_eff_abs) == min(η_tan_unbounded, η_eff_abs)
    @test rsf_preconditioner_tangent(η_eff_abs * 4, η_eff_abs) == η_eff_abs
    @test rsf_preconditioner_tangent(η_eff_abs / 4, η_eff_abs) == η_eff_abs / 4
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
    stokes.V.Vx .= PTArray(backend)(
        [
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi[1], y in yVx
        ]
    )
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5)
    rsf_bundle = (; ctrl = ctrl, fields = rsf_fields)

    # A soft tangent moves the diagonal when γ is comparable to η, and does not when γ dominates.
    γ_saved = copy(Array(dyrel.γ_eff))
    Dx_saved = copy(Array(dyrel.Dx))
    Dy_saved = copy(Array(dyrel.Dy))
    λx_saved = copy(Array(dyrel.λmaxVx))
    λy_saved = copy(Array(dyrel.λmaxVy))
    η_saved = copy(Array(stokes.viscosity.η))
    ηv_saved = copy(Array(stokes.viscosity.ηv))
    @test Array(rsf_fields.rsf_mask_v)[8, 8] > 0.5
    fill!(dyrel.γ_eff, 1.0)
    fill!(stokes.viscosity.η, 10.0)
    fill!(stokes.viscosity.ηv, 10.0)
    fill!(rsf_fields.η_tan, 0.0)
    fill!(rsf_fields.η_tan_v, 0.0)
    JustRelax2D.Gershgorin_Stokes2D_SchurComplement!(
        dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
        stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff,
        phase_ratios, rheology, grid.di, dt,
    )
    Dx_plain = copy(Array(dyrel.Dx))
    ηtv = zeros(size(Array(rsf_fields.η_tan_v)))
    ηtv[8, 8] = 1.0e-4
    copyto!(rsf_fields.η_tan_v, ηtv)
    JustRelax2D.Gershgorin_Stokes2D_SchurComplement!(
        dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
        stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff,
        phase_ratios, rheology, grid.di, dt,
        rsf_fields.η_tan, rsf_fields.η_tan_v, rsf_fields.rsf_mask, rsf_fields.rsf_mask_v,
    )
    @test !(Array(dyrel.Dx)[7, 8] ≈ Dx_plain[7, 8])
    fill!(dyrel.γ_eff, 1.0e12)
    ηtv[8, 8] = 10.0
    copyto!(rsf_fields.η_tan_v, ηtv)
    JustRelax2D.Gershgorin_Stokes2D_SchurComplement!(
        dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
        stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff,
        phase_ratios, rheology, grid.di, dt,
        rsf_fields.η_tan, rsf_fields.η_tan_v, rsf_fields.rsf_mask, rsf_fields.rsf_mask_v,
    )
    Dx_stiff = copy(Array(dyrel.Dx))
    ηtv[8, 8] = 1.0e-8
    copyto!(rsf_fields.η_tan_v, ηtv)
    JustRelax2D.Gershgorin_Stokes2D_SchurComplement!(
        dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
        stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff,
        phase_ratios, rheology, grid.di, dt,
        rsf_fields.η_tan, rsf_fields.η_tan_v, rsf_fields.rsf_mask, rsf_fields.rsf_mask_v,
    )
    @test isapprox(Array(dyrel.Dx)[7, 8], Dx_stiff[7, 8]; rtol = 1.0e-4)
    # γ ≫ η_tan, so λmax does not move; the pseudo-step is scaled by √(η_new/η_old).
    JustRelax2D.update_dτV_α_β!(dyrel)
    dτ_before = Array(dyrel.dτVx)[7, 8]
    @test dτ_before > 0
    η_prev_h = zeros(size(ηtv))
    η_prev_h[8, 8] = 1.0e-6
    η_prev = similar(rsf_fields.η_tan_v)
    copyto!(η_prev, η_prev_h)
    JustRelax2D.shrink_dτ_where_penalty_dominates!(dyrel, rsf_fields.η_tan_v, η_prev, rsf_fields.rsf_mask_v, 10.0)
    @test Array(dyrel.dτVx)[7, 8] ≈ dτ_before * 0.1 rtol = 1.0e-8
    # A drop of many decades is capped at the per-rebuild floor (default 0.1) ...
    JustRelax2D.update_dτV_α_β!(dyrel)
    @test Array(dyrel.dτVx)[7, 8] ≈ dτ_before rtol = 1.0e-8
    η_prev_h[8, 8] = 1.0e20
    copyto!(η_prev, η_prev_h)
    JustRelax2D.shrink_dτ_where_penalty_dominates!(dyrel, rsf_fields.η_tan_v, η_prev, rsf_fields.rsf_mask_v, 10.0)
    @test Array(dyrel.dτVx)[7, 8] ≈ dτ_before * 0.1 rtol = 1.0e-8
    # ... and the floor is a parameter.
    JustRelax2D.update_dτV_α_β!(dyrel)
    JustRelax2D.shrink_dτ_where_penalty_dominates!(dyrel, rsf_fields.η_tan_v, η_prev, rsf_fields.rsf_mask_v, 10.0, 1.0e-3)
    @test Array(dyrel.dτVx)[7, 8] ≈ dτ_before * 1.0e-3 rtol = 1.0e-8
    # α/β stay consistent with the scaled dτ.
    dτ78 = Array(dyrel.dτVx)[7, 8]
    c78 = Array(dyrel.cVx)[7, 8]
    @test Array(dyrel.βVx)[7, 8] ≈ 2 * dτ78 / (2 + c78 * dτ78)
    @test Array(dyrel.αVx)[7, 8] ≈ (2 - c78 * dτ78) / (2 + c78 * dτ78)
    copyto!(dyrel.γ_eff, γ_saved)
    copyto!(dyrel.Dx, Dx_saved)
    copyto!(dyrel.Dy, Dy_saved)
    copyto!(dyrel.λmaxVx, λx_saved)
    copyto!(dyrel.λmaxVy, λy_saved)
    copyto!(stokes.viscosity.η, η_saved)
    copyto!(stokes.viscosity.ηv, ηv_saved)
    fill!(rsf_fields.η_tan, 0.0)
    fill!(rsf_fields.η_tan_v, 0.0)

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
            rsf_tangent = true,
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

    # Second step with the relative floor, the relaxed fold and a custom shrink floor
    # threaded through the solver. The fault is still locked, so the converged state is the
    # same elastic loading: η_eff ≈ G dt sits above G dt / η_contrast, and the relaxed fold
    # has the Maxwell viscosity as both start and fixed point.
    out2 = solve_DYREL!(
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
            rsf_tangent = true,
            rsf_η_contrast = 1.0e4,
            rsf_relaxation = 0.5,
            rsf_shrink_min = 0.3,
        )
    )
    @test isfinite(out2.dt_rsf) && out2.dt_rsf > 0
    @test isapprox(maximum(abs, Array(stokes.τ.xy)), 2 * τ_elastic; rtol = 0.15)
    @test out2.Vp_max < 1.0e-3 * V_top
    @test minimum(Array(stokes.viscosity.η)) > 1.0e22
    @test minimum(Array(stokes.viscosity.ηv)) > 1.0e22
    # relaxation < 1 needs the history arrays when called directly
    @test_throws ErrorException apply_rsf_viscosity!(
        stokes, rsf_fields, ctrl, phase_ratios, dt; relaxation = 0.5
    )

    finalize_global_grid(; finalize_MPI = false)
end
