# Vendored from RheologyCalculator.jl + LaMEM constEq.cpp RSF timestep.
# Field naming follows LaMEM: Wf = fault width, L = D_rs (characteristic slip).

"""
    RateStateFriction{T}

Rate-and-state frictional viscosity (aging law in Ω = log Θ form).

# Fields (LaMEM names)
- `λ`   — fluid pressure ratio
- `μ₀`  — reference friction at `V₀`
- `V₀`  — reference slip velocity
- `a`, `b` — RSF constitutive parameters
- `L`   — characteristic slip distance (LaMEM `D_rs`)
- `C`   — cohesion
- `Wf`  — fault-zone width (`Vp = 2 Wf ε` in the constitutive branch)
"""
struct RateStateFriction{T}
    λ::T
    μ₀::T
    V₀::T
    a::T
    b::T
    L::T
    C::T
    Wf::T
end

RateStateFriction(args...) = RateStateFriction(promote(args...)...)

@inline _Vp_from_ε(r::RateStateFriction, ε) = 2 * r.Wf * ε

"""
LaMEM `updatePhaseRSFState` slip rate from stress (uses frozen Ω for the exp term):

``Vp = 2 V₀ sinh(max(τ-C,0)/(a P (1-λ))) exp(-(μ₀ + b Ω)/a)``
"""
@inline function compute_Vp_from_stress(
        r::RateStateFriction; τ = 0, Ω = 0, P = 0, kwargs...
    )
    Peff = max(P * (1 - r.λ), 1.0e-30)
    return 2 * r.V₀ * sinh(max(τ - r.C, 0) / (r.a * Peff)) * exp(-(r.μ₀ + r.b * Ω) / r.a)
end

# strain rate as a function of stress, state, and pressure
@inline function compute_strain_rate(r::RateStateFriction; τ = 0, Ω_old = 0, P = 0, dt = 0, kwargs...)
    Vp = compute_Vp_from_stress(r; τ = τ, Ω = Ω_old, P = P)
    return Vp / (2 * r.Wf)
end

# stress as a function of strain rate; updates Ω internally (RC default)
@inline function compute_stress(r::RateStateFriction; ε = 0, Ω_old = 0, P = 0, dt = 0, kwargs...)
    Vp = _Vp_from_ε(r, ε)
    Ω = update_Ω(r; ε = ε, Ω_old = Ω_old, dt = dt)
    μd = r.a * asinh(Vp / (2 * r.V₀) * exp((r.μ₀ + r.b * Ω) / r.a))
    τII = P * (1 - r.λ) * μd + r.C
    return τII
end

"""
    compute_stress_frozen_Ω(r; ε, Ω_old, P)

Stress from the asinh RSF law using a **frozen** state `Ω_old` (no `update_Ω`).
Use inside APT / PT iterations. Matches LaMEM `setupPhaseRSF`.
"""
@inline function compute_stress_frozen_Ω(r::RateStateFriction; ε = 0, Ω_old = 0, P = 0, kwargs...)
    Vp = _Vp_from_ε(r, ε)
    μd = r.a * asinh(Vp / (2 * r.V₀) * exp((r.μ₀ + r.b * Ω_old) / r.a))
    τII = P * (1 - r.λ) * μd + r.C
    return τII
end

# Aging-law state update (LaMEM). Θ = exp(Ω) floored before log.
# Vp here is from strain (constitutive); LaMEM post-solve path uses stress-based Vp.
@inline function update_Ω(r::RateStateFriction; ε = 0, Ω_old = 0, dt = 0, Vp = nothing, kwargs...)
    Vp_ = isnothing(Vp) ? _Vp_from_ε(r, ε) : Vp
    if (Vp_ * dt / r.L ≤ 1.0e-6)
        Θ = exp(Ω_old) * (1 - Vp_ * dt / r.L) + r.V₀ * dt / r.L
    else
        Θ = r.V₀ / Vp_ + (exp(Ω_old) - r.V₀ / Vp_) * exp(-Vp_ * dt / r.L)
    end
    return log(max(Θ, 1.0e-300))
end

# ---- Adaptive timestep — LaMEM constEq.cpp `updatePhaseRSFState` ----

"""
Lapusta / LaMEM ``dteta_max`` ∈ [0.1, 0.2] from elastic stiffness ``k = 2/π · G/((1-ν) Wf)``.
Uses ``3.14`` for π as in LaMEM.
"""
@inline function dteta_max_lamem(a, b, L, Wf, G, ν, dP)
    dP = max(dP, 1.0e-30)
    k = (2 / 3.14) * ((G / (1 - ν)) / Wf)
    xi = 0.25 * ((k * L) / (a * dP) - (b - a) / a)^2 - (k * L) / (a * dP)
    dteta = if xi > 0
        a * dP / (k * L - (b - a) * dP)
    else
        1 - (b - a) * dP / (k * L)
    end
    dteta = min(dteta, 0.2)
    return dteta < 0.1 ? 0.1 : dteta
end

"""Healing (LaMEM): ``0.2 D_rs / (V₀ e^{-Ω})`` — uses **updated** state."""
@inline function dt_healing(; L, V₀, Ω, kwargs...)
    return 0.2 * L / (V₀ * exp(-Ω))
end
# keep r-based overload for unit tests
@inline function dt_healing(r::RateStateFriction; Ω_old = 0, kwargs...)
    return dt_healing(; L = r.L, V₀ = r.V₀, Ω = Ω_old)
end

"""Weakening (LaMEM): ``min(dteta_max · D_rs / Vp, 1e9)``."""
@inline function dt_weakening(; L, Vp, dteta_max = 0.2, kwargs...)
    return min(dteta_max * L / max(Vp, 1.0e-30), 1.0e9)
end
@inline function dt_weakening(r::RateStateFriction; ε = 0, kwargs...)
    return dt_weakening(; L = r.L, Vp = _Vp_from_ε(r, ε), dteta_max = 0.2)
end

"""Slip Courant (LaMEM): ``1e-3 · Wf / Vp``."""
@inline function dt_courant(; Wf, Vp, f = 1.0e-3, kwargs...)
    return f * Wf / max(Vp, 1.0e-30)
end
@inline function dt_courant(r::RateStateFriction; ε = 0, f = 1.0e-3, kwargs...)
    return dt_courant(; Wf = r.Wf, Vp = _Vp_from_ε(r, ε), f = f)
end

"""
    compute_dt_ratestate_lamem(; Wf, L, V₀, a, b, G, ν, P, Vp, Ω)

Full LaMEM cell timestep: ``min(dt_w, dt_h, dt_c, 1e9)``.
"""
@inline function compute_dt_ratestate_lamem(; Wf, L, V₀, a, b, G, ν, P, Vp, Ω)
    dP = max(abs(P), 1.0e-30)
    dtw = dt_weakening(; L = L, Vp = Vp, dteta_max = dteta_max_lamem(a, b, L, Wf, G, ν, dP))
    dth = dt_healing(; L = L, V₀ = V₀, Ω = Ω)
    dtc = dt_courant(; Wf = Wf, Vp = Vp)
    return min(dtw, dth, dtc, 1.0e9)
end

"""Backward-compatible RC-style helper (fixed dteta_max=0.2, Vp from ε). """
@inline function compute_dt_ratestate(
        r::RateStateFriction;
        ε = 0,
        Ω_old = 0,
        dt = 0,
        f = 1.0e-3,
        kwargs...,
    )
    Vp = _Vp_from_ε(r, ε)
    return min(
        dt_healing(r; Ω_old = Ω_old),
        dt_weakening(r; ε = ε),
        dt_courant(r; ε = ε, f = f),
        1.0e9,
    )
end

@inline function max_state_change(r::RateStateFriction; ε = 0, Ω_old = 0, dt = 0, kwargs...)
    Ω = update_Ω(r; ε = ε, Ω_old = Ω_old, dt = dt)
    return abs(Ω - Ω_old)
end
