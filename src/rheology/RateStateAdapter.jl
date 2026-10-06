# Thin adapter between LaMEM / JustRelax RSF names and a future RheologyCalculator
# `RateStateFriction` law. DYREL and kernels call these helpers so the constitutive
# backend can be swapped without touching the solver loop.
#
# Name map (LaMEM / JR → RheologyCalculator):
#   a_rsf, b_rsf → a, b
#   mu0_rsf      → μ₀
#   V0           → V₀
#   D_rs         → L
#   Wf           → D  (fault / cell width; Vp = 2 Wf ε)
#   λ, C         → λ, C

"""
    rsf_viscosity_from_stress(r; τ, Ω_old, P) -> (η_rsf, Vp)

Isolated RSF dashpot viscosity from stress (frozen Ω):
``η_rsf = τ Wf / Vp`` with ``Vp`` from [`compute_Vp_from_stress`](@ref).
``Inf`` when locked. Used for LaMEM-style η bounds / guesses, not as the final η.
"""
@inline function rsf_viscosity_from_stress(
        r::RateStateFriction; τ = 0, Ω_old = 0, P = 0, ε_floor = 1.0e-30, kwargs...
    )
    Vp = compute_Vp_from_stress(r; τ = τ, Ω = Ω_old, P = P)
    if !(isfinite(Vp)) || Vp ≤ ε_floor || !(isfinite(τ)) || τ ≤ ε_floor
        return Inf, max(Vp, zero(Vp))
    end
    η_rsf = τ * r.Wf / Vp
    return η_rsf, Vp
end

"""
    rsf_cons_eq_residual(η, DII, η_creep, Gdt, r, Ω_old, P)

LaMEM `getConsEqRes` for Maxwell creep + elasticity + RSF (frozen Ω):

```
τ = 2 η DII
r = DII - [τ/(2 G dt) + τ/(2 η_creep) + Vp(τ)/(2 Wf)]
```
with ``G dt ≡ Gdt``. Negative when `η` is too large.
"""
@inline function rsf_cons_eq_residual(η, DII, η_creep, Gdt, r::RateStateFriction, Ω_old, P)
    τ = 2 * η * DII
    DIIels = τ / (2 * Gdt)
    DIIcreep = τ / (2 * max(η_creep, eps(typeof(η_creep))))
    Vp = compute_Vp_from_stress(r; τ = τ, Ω = Ω_old, P = P)
    DIIrsf = Vp / (2 * r.Wf)
    return DII - (DIIels + DIIcreep + DIIrsf)
end

"""
    solve_bisect_eta_rsf(η_lo, η_hi, tol, maxit, DII, η_creep, Gdt, r, Ω_old, P) -> η

LaMEM `solveBisect` on [`rsf_cons_eq_residual`](@ref). Fixed iteration budget (GPU-safe).
Returns the lower-bound guess if the residual does not change sign.
"""
@inline function solve_bisect_eta_rsf(
        η_lo, η_hi, tol, maxit, DII, η_creep, Gdt, r::RateStateFriction, Ω_old, P
    )
    a = η_lo
    b = η_hi
    x = a
    fa = rsf_cons_eq_residual(a, DII, η_creep, Gdt, r, Ω_old, P)
    abs(fa) ≤ tol && return a
    fb = rsf_cons_eq_residual(b, DII, η_creep, Gdt, r, Ω_old, P)
    # no bracket → closed-form / harmonic guess
    (fa * fb > 0) && return a
    @inbounds for _ in 1:maxit
        x = (a + b) * 0.5
        fx = rsf_cons_eq_residual(x, DII, η_creep, Gdt, r, Ω_old, P)
        abs(fx) ≤ tol && return x
        if fa * fx < 0
            b = x
        else
            a = x
            fa = fx
        end
    end
    return x
end

"""
    rsf_effective_viscosity(r; DII, η_creep, Gdt, Ω_old, P, kwargs...) -> (η_eff, η_creep_out, Vp, τ)

LaMEM cell solve: bisection for effective ``η`` with ``τ = 2 η DII``, then convert to the
JustRelax Maxwell **creep** buffer so ``η_ve(η_creep_out) = η_eff``:

```
η_ve = 1 / (1/η_creep + 1/Gdt)
```
"""
@inline function rsf_effective_viscosity(
        r::RateStateFriction;
        DII,
        η_creep,
        Gdt,
        Ω_old = 0,
        P = 0,
        tol_rel = 1.0e-12,
        maxit = 50,
        ε_floor = 1.0e-30,
        kwargs...,
    )
    DII = max(DII, ε_floor)
    Gdt = max(Gdt, ε_floor)
    η_creep = max(η_creep, ε_floor)

    # Bounds (LaMEM): harmonic lower bound, softest single-mechanism upper bound.
    # Use stress-based η_rsf at the Maxwell trial τ so locked cells contribute inv_rsf ≈ 0
    # (asinh-at-DII would spuriously soften the bracket while DIIrsf(τ) is still ~0).
    inv_els = inv(Gdt)
    inv_creep = inv(η_creep)
    η_ve0 = inv(inv_els + inv_creep)
    τ_trial = 2 * η_ve0 * DII
    η_rsf_g, _ = rsf_viscosity_from_stress(r; τ = τ_trial, Ω_old = Ω_old, P = P, ε_floor = ε_floor)
    inv_rsf = isfinite(η_rsf_g) && η_rsf_g > ε_floor ? inv(η_rsf_g) : zero(η_creep)

    inv_sum = inv_els + inv_creep + inv_rsf
    η_lo = inv(max(inv_sum, ε_floor))                     # harmonic mean
    inv_hi = max(inv_els, inv_creep, inv_rsf)
    η_hi = inv(max(inv_hi, ε_floor))                      # softest mechanism
    # ensure lo ≤ hi
    if η_lo > η_hi
        η_lo, η_hi = η_hi, η_lo
    end

    tol = tol_rel * DII
    η_eff = solve_bisect_eta_rsf(η_lo, η_hi, tol, maxit, DII, η_creep, Gdt, r, Ω_old, P)
    η_eff = min(max(η_eff, η_lo), η_hi)

    τ = 2 * η_eff * DII
    Vp = compute_Vp_from_stress(r; τ = τ, Ω = Ω_old, P = P)

    # Map LaMEM η_eff → JR creep η so Maxwell update reproduces η_ve = η_eff.
    # If η_eff matches Maxwell-only η_ve0, keep the GeoParams buffer unchanged.
    η_creep_out = if η_eff ≥ η_ve0 * (1 - 1.0e-8)
        η_creep
    else
        inv_c = inv(η_eff) - inv(Gdt)
        inv_c > ε_floor ? inv(inv_c) : η_creep
    end

    return η_eff, η_creep_out, Vp, τ
end

"""
    rsf_frozen_stress_viscosity(r; ε, Ω_old, P) -> (τ_rsf, η_rsf, Vp)

Asinh / strain-rate branch (RheologyCalculator `compute_stress` style).
Prefer [`rsf_effective_viscosity`](@ref) inside DYREL / PT iterations.
"""
@inline function rsf_frozen_stress_viscosity(
        r::RateStateFriction; ε = 0, Ω_old = 0, P = 0, kwargs...
    )
    τ_rsf = compute_stress_frozen_Ω(r; ε = ε, Ω_old = Ω_old, P = P)
    Vp = 2 * r.Wf * ε
    η_rsf = τ_rsf / (2 * max(ε, 1.0e-30))
    return τ_rsf, η_rsf, Vp
end

"""
    rsf_from_phase_params(p::PhaseRSFParams, a, b) -> RateStateFriction

Build a local constitutive object, overriding `a`/`b` with spatially evaluated values.
"""
@inline function rsf_from_phase_params(p::PhaseRSFParams, a, b)
    return RateStateFriction(p.λ, p.μ₀, p.V₀, a, b, p.L, p.C, p.Wf)
end

"""
    to_rheology_calculator_kwargs(r::RateStateFriction)

Keyword bag with RheologyCalculator field names for a future backend swap.
"""
@inline function to_rheology_calculator_kwargs(r::RateStateFriction)
    return (;
        λ = r.λ,
        μ₀ = r.μ₀,
        V₀ = r.V₀,
        a = r.a,
        b = r.b,
        L = r.L,
        C = r.C,
        D = r.Wf, # RC uses `D` for fault width
    )
end
