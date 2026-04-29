# Unit/regression tests for the DYREL RSF branch (`_compute_local_stress`): sinh slip velocity at
# converged τII, Ω update, viscoplastic η cap, and a short multistep march with simple-shear τ memory.
# For RheologyCalculator ↔ JustRelax coupled 0D integration (same driver as the RC example), see
# `test_rsf_rc_0d_snapshots.jl`.

using Test
using JustRelax
using GeoParams: second_invariant

# Mirrors `JustRelax2D._rsf_vp_sinh_ratestate` / RheologyCalculator slip law (for regression checks).
function _vp_sinh_ref(τII, Ω_old, p, rsf, phase::Int = 1)
    a = rsf.a[phase]
    b = rsf.b[phase]
    μ0 = rsf.μ0[phase]
    V0 = rsf.V0[phase]
    λ_rsf = rsf.λ[phase]
    C_rsf = rsf.C[phase]
    Vp_max = rsf.Vp_max[phase]
    denom = max(a * p * (1.0 - λ_rsf), eps(Float64))
    x = max(τII - C_rsf, 0.0) / denom
    x = min(x, 350.0)
    ex = exp(-(μ0 + b * Ω_old) / max(a, eps(Float64)))
    vp = 2.0 * V0 * sinh(x) * ex
    return clamp(vp, 0.0, Vp_max)
end

function _omega_ref(Ω_old, vp, dt, rsf, phase::Int = 1)
    L = rsf.L[phase]
    V0 = rsf.V0[phase]
    var_rsf = (vp * dt) / max(L, eps(Float64))
    Ω = if var_rsf <= 1.0e-6
        log(exp(Ω_old) * (1.0 - var_rsf) + V0 * dt / max(L, eps(Float64)))
    else
        log(V0 / max(vp, eps(Float64)) + (exp(Ω_old) - V0 / max(vp, eps(Float64))) * exp(-var_rsf))
    end
    return clamp(Ω, -100.0, 100.0)
end

@testset "RSF local solver (0D-style sanity)" begin
    # Single material-point setup (mirrors the 0D rate-state workflow)
    eij = (1.5e-14, -1.5e-14, 0.0)
    tau_old = (0.0, 0.0, 0.0)
    eta = 1.0e22
    p = 50.0e6
    g = 20.0e9
    kb = 50.0e9
    lambda_old = 0.0
    lambda_relax = 1.0
    dt = 5.0e8
    omega_old = -3.0

    # Plastic branch is off; RSF is used as viscous weakening law.
    is_plastic = false
    cohesion_dp = 0.0
    sinphi = 0.0
    cosphi = 1.0
    sinpsi = 0.0
    eta_reg = 0.0

    # Same solver options as `RheologyCalculator.jl/examples/RateState_compare_JustRelax.jl`
    rsf = (
        active = (true,),
        a = (0.011,),
        b = (0.015,),
        μ0 = (0.5,),
        V0 = (4.0e-9,),
        L = (0.0047,),
        D = (500.0,),
        λ = (0.0,),
        C = (0.0,),
        η_min = (1.0e18,),
        η_max = (1.0e24,),
        Vp_max = (1.0e19,),
        p_shift = (0.0,),
        use_bisection = (false,),
        maxit = (12,),
        rtol = (1.0e-6,),
    )

    tau_xx, tau_yy, tau_xy, epl_xx, epl_yy, epl_xy, tau_ii, lambda_new, dppsi, eta_vep, omega_new, vp =
        JustRelax.JustRelax2D._compute_local_stress(
            eij,
            tau_old,
            eta,
            p,
            g,
            kb,
            lambda_old,
            lambda_relax,
            is_plastic,
            cohesion_dp,
            sinphi,
            cosphi,
            sinpsi,
            eta_reg,
            dt,
            rsf,
            1,
            omega_old,
        )

    @test isfinite(tau_ii)
    @test isfinite(eta_vep)
    @test isfinite(omega_new)
    @test vp >= 0.0
    @test vp <= rsf.Vp_max[1]

    eta_ve = (eta * g * dt) / (eta + g * dt)
    inv_2gdt = inv(2 * g * dt)
    eij_eff = @. eij + tau_old * inv_2gdt
    eii_eff = second_invariant(eij_eff)

    @test eta_vep <= eta_ve * (1.0 + 1.0e-12)
    @test eta_vep >= 0.0
    @test isapprox(tau_ii, 2.0 * eta_vep * eii_eff; rtol = 1.0e-6, atol = 1.0)

    vp_ref = _vp_sinh_ref(tau_ii, omega_old, p, rsf)
    @test isapprox(vp, vp_ref; rtol = 1.0e-10, atol = 1.0e-10)

    omega_ref = _omega_ref(omega_old, vp, dt, rsf)
    @test isapprox(omega_new, omega_ref; rtol = 1.0e-10, atol = 1.0e-10)

    # RSF branch should not generate DP plastic strain or pressure correction.
    @test epl_xx == 0.0
    @test epl_yy == 0.0
    @test epl_xy == 0.0
    @test lambda_new == 0.0
    @test dppsi == 0.0
    @test isfinite(tau_xx) && isfinite(tau_yy) && isfinite(tau_xy)
end

@testset "RSF multistep: simple-shear τ memory + Ω chain (0D coupled pattern)" begin
    # Same physics as `stress_time_coupled`: τ_hist = (τ_prev, -τ_prev, 0), Ω_old from previous step.
    eij = (1.5e-14, -1.5e-14, 0.0)
    eta = 1.0e22
    p = 50.0e6
    g = 20.0e9
    kb = 50.0e9
    dt = 5.0e8
    omega_hist = -3.0
    τ_mem = 0.0

    rsf = (
        active = (true,),
        a = (0.011,),
        b = (0.015,),
        μ0 = (0.5,),
        V0 = (4.0e-9,),
        L = (0.0047,),
        D = (500.0,),
        λ = (0.0,),
        C = (0.0,),
        η_min = (1.0e18,),
        η_max = (1.0e24,),
        Vp_max = (1.0e19,),
        p_shift = (0.0,),
        use_bisection = (false,),
        maxit = (12,),
        rtol = (1.0e-6,),
    )

    for _ in 1:20
        tau_hist = (τ_mem, -τ_mem, 0.0)
        tx, ty, txy, _, _, _, tii, _, _, _, omega_step, vp_step =
            JustRelax.JustRelax2D._compute_local_stress(
                eij,
                tau_hist,
                eta,
                p,
                g,
                kb,
                0.0,
                1.0,
                false,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                dt,
                rsf,
                1,
                omega_hist,
            )
        @test isfinite(tii)
        @test isfinite(omega_step)
        @test vp_step >= 0.0
        @test vp_step <= rsf.Vp_max[1]
        # Pure shear: σyy ≈ -σxx
        @test isapprox(ty, -tx; rtol = 1.0e-5, atol = 1.0e-8 * max(abs(tx), 1.0))
        @test abs(txy) <= 1.0e-8 * max(abs(tx), 1.0)

        @test isapprox(vp_step, _vp_sinh_ref(tii, omega_hist, p, rsf); rtol = 1.0e-9, atol = 1.0e-9)
        @test isapprox(omega_step, _omega_ref(omega_hist, vp_step, dt, rsf); rtol = 1.0e-9, atol = 1.0e-9)

        τ_mem = tx
        omega_hist = omega_step
    end
end
