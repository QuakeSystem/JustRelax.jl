# Elastic eigenstrain rate-and-state (Garnet / SEAS split).
#
# The momentum solve stays elastic. Slip on one fault row is an eigenstrain
# `V / (2 Wf)`, not a frictional viscosity. `Wf` is the LaMEM fault width
# (zone thickness, usually 25–500 m). It is not the SEAS down-dip fault height
# (`Wf_height`; BP1-FD calls that length `Wf` as well).
# Slip rate and aging reuse `RateStateFriction`.

"""
    eigenstrain_trial_stress(h, ε, V_lag, G, dt, Wf) -> τ

Elastic fault traction with a frozen slip rate:

```
τ = h + 2 G Δt (ε − sign(τ_el) V_lag / (2 Wf))
```

`τ_el = h + 2 G Δt ε` is the traction before slip. The result stays on the same
side of zero as `τ_el` and never overshoots it. A zero elastic traction is left
at zero (no eigenstrain).
"""
@inline function eigenstrain_trial_stress(h, ε, V_lag, G, dt, Wf)
    τ_el = h + 2 * G * dt * ε
    τ_el == 0 && return zero(τ_el)
    s = ifelse(τ_el > 0, one(τ_el), -one(τ_el))
    ė_slip = s * V_lag / (2 * Wf)
    τ_new = h + 2 * G * dt * (ε - ė_slip)
    return ifelse(
        τ_el > 0,
        min(τ_el, max(τ_new, zero(τ_el))),
        max(τ_el, min(τ_new, zero(τ_el))),
    )
end

"""Local `RateStateFriction` with nodal `a` and `b`. `Wf` on the prototype is unused by the slip law."""
@inline function eigenstrain_law(proto::RateStateFriction{T}, a, b) where {T}
    return RateStateFriction{T}(proto.λ, proto.μ₀, proto.V₀, convert(T, a), convert(T, b), proto.L, proto.C, proto.Wf)
end

"""
    eigenstrain_slip_rate(proto, a, b, τ, ψ, σ; V_cap=1000) -> V

Magnitude of the slip rate from the frozen-state law at effective normal stress `σ`.
Non-finite or non-positive rates become zero. `V` is capped at `V_cap` (LaMEM, 1000 m/s).
"""
@inline function eigenstrain_slip_rate(proto::RateStateFriction, a, b, τ, ψ, σ, V_cap)
    r = eigenstrain_law(proto, a, b)
    Vp = compute_Vp_from_stress(r; τ = abs(τ), Ω = ψ, P = σ)
    return ifelse(isfinite(Vp) && Vp > 0, min(Vp, convert(typeof(Vp), V_cap)), zero(Vp))
end

eigenstrain_slip_rate(proto, a, b, τ, ψ, σ; V_cap = 1000) =
    eigenstrain_slip_rate(proto, a, b, τ, ψ, σ, V_cap)

"""
    eigenstrain_age_state(proto, a, b, ψ_old, V, dt) -> ψ

One aging step. A non-positive slip rate heals with `Vp = 0`, which is the locked-fault
limit of the LaMEM aging law (`Θ` grows as `V₀ Δt / L`).
"""
@inline function eigenstrain_age_state(proto::RateStateFriction, a, b, ψ_old, V, dt)
    r = eigenstrain_law(proto, a, b)
    Vp = V > 0 ? V : zero(V)
    return update_Ω(r; Ω_old = ψ_old, dt = dt, Vp = Vp)
end

"""
Steady-state shear stress at slip rate `V_init` (cookbook `τ0`).

`ψ = ln(V0 / V_init)` is implied; pass the same `a` that multiplies `σ`.
"""
@inline function eigenstrain_steady_stress(a, σ, μ₀, b, V₀, V_init)
    return σ * a * asinh((V_init / (2 * V₀)) * exp((μ₀ + b * log(V₀ / V_init)) / a))
end

"""
`ψ` such that the slip law returns `V` at traction `τ` and effective stress `σ`.

Uses `ln(sinh x) ≈ x − ln 2` for `x > 20`.
"""
@inline function eigenstrain_ψ_for_slip(a, b, μ₀, V₀, V, τ, σ)
    x = τ / (a * σ)
    lnsinh = x > 20 ? x - log(2) : log(sinh(x))
    return (a * (log(2 * V₀ / V) + lnsinh) - μ₀) / b
end
