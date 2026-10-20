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

# Keyword Arguments
- `η_contrast`: Floor ``η_eff ≥ Gdt / η_contrast``, i.e. the node may not be softer than the
  elastic wall by more than this factor. Locked nodes (``η_eff ≈ Gdt``) are unaffected.
  `Inf` disables the floor. Default: `Inf`.
- `η_min`, `η_max`: Absolute bounds on the Maxwell **creep** buffer (same meaning as
  `viscosity_cutoff`). After the bisection, `η_eff` is raised/lowered to the Maxwell image
  of those bounds, ``1 / (1/η_bound + 1/Gdt)``, and ``τ = 2 η_eff DII``, `Vp` and the creep
  buffer are recomputed from that floored value. That way the stress the Stokes update
  will form from the clamped buffer matches the fold. Defaults: `-Inf`, `Inf`.
- `η_prev`, `relax`: Under-relax the fold toward the previous iterate in log space,
  ``η_eff = exp((1 - relax) log η_prev + relax log η_eff)``, when `relax < 1` and `η_prev`
  is a finite positive number. `τ`, `Vp` and the creep buffer are evaluated at the relaxed
  value, so the fixed point of repeated folds is unchanged. Default: `nothing`, `1`.
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
        η_contrast = Inf,
        η_min = -Inf,
        η_max = Inf,
        η_prev = nothing,
        relax = 1,
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
    η_rsf_g, Vp_g = rsf_viscosity_from_stress(r; τ = τ_trial, Ω_old = Ω_old, P = P, ε_floor = ε_floor)
    inv_rsf = isfinite(η_rsf_g) && η_rsf_g > ε_floor ? inv(η_rsf_g) : zero(η_creep)

    inv_sum = inv_els + inv_creep + inv_rsf
    η_lo = inv(max(inv_sum, ε_floor))                     # harmonic mean
    inv_hi = max(inv_els, inv_creep, inv_rsf)
    η_hi = inv(max(inv_hi, ε_floor))                      # softest mechanism
    # ensure lo ≤ hi
    if η_lo > η_hi
        η_lo, η_hi = η_hi, η_lo
    end
    # The Maxwell trial stress is so far past locking that `sinh` overflowed, or the
    # stress-based guess fell below ε_floor and was dropped from `inv_rsf` above. Either way
    # the bracket has just collapsed onto η_ve0 and would hand back the *elastic* viscosity
    # for the softest possible node. Bracket the whole range instead: the residual is
    # positive at ε_floor and very negative at η_ve0, so the bisection finds the real root.
    very_soft = τ_trial > ε_floor && (!isfinite(Vp_g) || (isfinite(η_rsf_g) && η_rsf_g ≤ ε_floor))
    if very_soft
        η_lo = ε_floor
        η_hi = η_ve0
    end

    tol = tol_rel * DII
    η_eff = solve_bisect_eta_rsf(η_lo, η_hi, tol, maxit, DII, η_creep, Gdt, r, Ω_old, P)
    η_eff = min(max(η_eff, η_lo), η_hi)

    # Relative floor: no softer than the elastic wall by more than η_contrast. The floor
    # cannot exceed η_ve0, so locked nodes are unchanged.
    if isfinite(η_contrast) && η_contrast > 0
        η_eff = max(η_eff, min(Gdt / η_contrast, η_ve0))
    end

    # Under-relax toward the previous iterate (log space: viscosities span decades).
    # Both ends are already floored, so the interpolant respects the floor.
    η_eff = _relax_rsf_viscosity(η_eff, η_prev, relax, η_ve0)

    # Absolute creep-buffer cutoff, applied on η_eff before τ/Vp so the Stokes stress
    # update (which Maxwell-combines the clamped buffer) sees the same η. Floor first:
    # harmonic(η_min, Gdt) is what η_ve becomes once η_creep is raised to η_min.
    η_eff = _clamp_rsf_effective(η_eff, η_min, η_max, Gdt, ε_floor)

    τ = 2 * η_eff * DII
    Vp = compute_Vp_from_stress(r; τ = τ, Ω = Ω_old, P = P)
    η_creep_out = _rsf_creep_from_effective(η_eff, η_ve0, η_creep, Gdt, ε_floor)
    # Keep the buffer inside the cutoff even when η_eff was already above the Maxwell
    # image (e.g. locked cell with a tiny η_creep that still maps to η_ve0 ≈ Gdt).
    η_creep_out = min(max(η_creep_out, η_min), η_max)

    return η_eff, η_creep_out, Vp, τ
end

# Maxwell image of a creep-buffer bound: η_ve(η) = 1/(1/η + 1/Gdt).
@inline function _maxwell_image(η_bound, Gdt, ε_floor)
    return inv(inv(max(η_bound, ε_floor)) + inv(Gdt))
end

@inline function _clamp_rsf_effective(η_eff, η_min, η_max, Gdt, ε_floor)
    if isfinite(η_min) && η_min > 0
        η_eff = max(η_eff, _maxwell_image(η_min, Gdt, ε_floor))
    end
    if isfinite(η_max) && η_max > 0
        η_eff = min(η_eff, _maxwell_image(η_max, Gdt, ε_floor))
    end
    return η_eff
end

@inline _relax_rsf_viscosity(η_eff, ::Nothing, relax, η_ve0) = η_eff

@inline function _relax_rsf_viscosity(η_eff, η_prev, relax, η_ve0)
    if relax < 1 && isfinite(η_prev) && η_prev > 0
        η_prev = min(η_prev, η_ve0)
        η_eff = continuation_log(η_eff, η_prev, relax)
    end
    return η_eff
end

# Map LaMEM η_eff → JR creep η so the Maxwell update reproduces η_ve = η_eff.
# If η_eff matches Maxwell-only η_ve0, keep the GeoParams buffer unchanged.
@inline function _rsf_creep_from_effective(η_eff, η_ve0, η_creep, Gdt, ε_floor)
    return if η_eff ≥ η_ve0 * (1 - 1.0e-8)
        η_creep
    else
        inv_c = inv(η_eff) - inv(Gdt)
        inv_c > ε_floor ? inv(inv_c) : η_creep
    end
end

"""
    rsf_tangent_viscosity(r; τ, Ω, P, η_creep, Gdt) -> η_tan

Viscosity seen by the Gershgorin row: the derivative of the frozen-state
constitutive law, not the secant `τ / (2ε)`.

```
η_tan = 1 / (1/(G dt) + 1/η_creep + (dVp/dτ)/Wf)
```

Locked faults (`dVp/dτ = 0`) reduce to the Maxwell viscosity. The value already
includes elasticity and creep, so a preconditioner must not Maxwell-combine it again.

When the secant `η_eff` has been raised (cutoff / contrast floor), Stokes iterates
that linear dashpot, not the unbounded RSF slope. [`rsf_preconditioner_tangent`](@ref)
then takes `min(η_tan, η_eff)` so the pseudo-step is sized for the operator that
actually updates stress.
"""
@inline function rsf_tangent_viscosity(
        r::RateStateFriction;
        τ,
        Ω = 0,
        P = 0,
        η_creep,
        Gdt,
        ε_floor = 1.0e-30,
        kwargs...,
    )
    Gdt = max(Gdt, ε_floor)
    η_creep = max(η_creep, ε_floor)
    inv_tan = inv(Gdt) + inv(η_creep)
    slope = dVp_dτ(r; τ = τ, Ω = Ω, P = P)
    if isfinite(slope) && slope > 0
        inv_tan += slope / r.Wf
    end
    η_tan = inv(inv_tan)
    if !(isfinite(η_tan)) || η_tan ≤ 0
        η_tan = inv(inv(Gdt) + inv(η_creep))
    end
    return η_tan
end

"""
    rsf_preconditioner_tangent(η_tan, η_eff) -> η

Slope written into the DYREL pseudo-step. Never stiffer than the secant Stokes
actually uses: a cutoff/contrast floor raises `η_eff` to a linear dashpot, and the
unbounded RSF derivative at that stress can sit *above* it (this run: `η_tan ≈ 3e9`
with `η_eff = 5e8`). Sizing `dτ` from the stiffer number leaves the residual on the
soft cells, rebuilds every inner iteration, and trips `GrowingPH`.
"""
@inline rsf_preconditioner_tangent(η_tan, η_eff) = min(η_tan, η_eff)

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
