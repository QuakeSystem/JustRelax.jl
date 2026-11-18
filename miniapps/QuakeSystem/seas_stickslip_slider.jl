# Scalar stick-slip slider with the SEAS BP1-FD velocity-weakening numbers.
#
# The spring stiffness is `k = G / H`, with `H` the fault-normal distance between
# the locked wall and the plate. `k_crit = σ (b − a) / L`. A fault cell of width
# `Wf = 100 m` on its own has `G / Wf ≈ 10 k_crit`, so it creeps. Stick-slip needs
# `H` larger than `G / k_crit ≈ 1.0 km`. Default `H = 4 km`.
#
# Each step freezes `V`, updates the spring `τ ← τ + k Δt (Vp − V)`, subtracts
# radiation damping `η V` with `η = G / (2 cs)`, inverts the sinh law, and ages
# once. The damping is what inertia does in DYREL; without it this quasi-static
# spring runs away to `V_cap`. Same pass count and tolerance as
# `eigenstrain_physical_step!`.
#
# Included by the stick-slip miniapp and by `test/test_dyrel_eigenstrain_stickslip_2D.jl`.
# Requires `JustRelax` to be loaded.

"""SEAS BP1-FD velocity-weakening slider. `Δψ > 0` starts stronger than steady sliding."""
function seas_bp1_vw_params(; Δψ = 2.0, H = 4.0e3)
    Wf = 100.0
    G = 3.2038e10
    ν = 0.25
    ρ = 2670.0
    η = 5.0e26
    σ = 50.0e6
    a = 0.01
    b = 0.015
    μ₀ = 0.6
    V₀ = 1.0e-6
    L = 0.008
    Vp = 1.0e-9
    V_cap = 1000.0
    tol = 1.0e-2
    # Four passes is what the BP1 miniapp uses. During the acceleration that
    # killed that run the rate is still moving after four, so the probe allows eight.
    max_pass = 8
    proto = RateStateFriction(0.0, μ₀, V₀, a, b, L, 0.0, Wf)
    τ_ss = eigenstrain_steady_stress(a, σ, μ₀, b, V₀, Vp)
    ψ_ss = eigenstrain_ψ_for_slip(a, b, μ₀, V₀, Vp, τ_ss, σ)
    ψ = ψ_ss + Δψ
    τ = τ_ss
    V = eigenstrain_slip_rate(proto, a, b, τ, ψ, σ, V_cap)
    k = G / H
    k_fault = G / Wf
    k_crit = σ * (b - a) / L
    η_rad = G / (2 * sqrt(G / ρ))
    # Stress may rise by this fraction of `a σ` per step while the fault is locked.
    # Without it, a healing step from a stuck fault is ~1e8–1e9 s and the next
    # fixed-point pass unloads the fault in one go.
    dt_load = 0.05 * a * σ / (k * Vp)
    return (;
        Wf, H, G, ν, ρ, η, σ, a, b, μ₀, V₀, L, Vp, V_cap, tol, max_pass,
        proto, τ_ss, ψ_ss, τ, ψ, V, dt_load, k, k_fault, k_crit, η_rad, Δψ,
    )
end

"""Slip rate the law returns when the spring and radiation damping are at `Vs`."""
function _slider_rate(p, τ, ψ, dt, Vs)
    τ_spring = τ + p.k * dt * (p.Vp - Vs)
    τ_fault = τ_spring - p.η_rad * Vs
    # `eigenstrain_slip_rate` takes `|τ|`. A negative fault stress must be a
    # locked fault, not a second high-rate branch.
    τ_fault > 0 || return 0.0
    return eigenstrain_slip_rate(p.proto, p.a, p.b, τ_fault, ψ, p.σ, p.V_cap)
end

"""
One accepted slider step.

The rate residual `V - law(τ(V))` is monotone, so the root is bracketed instead of
iterated. Picard iteration on this residual oscillates once the fault is fast,
which is the same four-pass loop DYREL uses.
"""
function slider_step(p, τ, ψ, V, dt)
    if _slider_rate(p, τ, ψ, dt, p.V_cap) >= p.V_cap
        V_new = p.V_cap
    else
        lo, hi = 0.0, p.V_cap
        for _ in 1:50
            mid = 0.5 * (lo + hi)
            _slider_rate(p, τ, ψ, dt, mid) > mid ? (lo = mid) : (hi = mid)
        end
        V_new = 0.5 * (lo + hi)
    end
    rel = abs(V_new - _slider_rate(p, τ, ψ, dt, V_new)) / max(V_new, eps())
    τ_spring = τ + p.k * dt * (p.Vp - V_new)
    ψ_new = eigenstrain_age_state(p.proto, p.a, p.b, ψ, V_new, dt)
    return (; τ = τ_spring, ψ = ψ_new, V = V_new, passes = 1, rel = rel)
end

"""LaMEM rate-and-state step, capped so locked loading cannot jump the strength."""
function slider_dt(p, V, ψ)
    dt = compute_dt_ratestate_lamem(;
        Wf = p.Wf, L = p.L, V₀ = p.V₀, a = p.a, b = p.b,
        G = p.G, ν = p.ν, P = p.σ, Vp = V, Ω = ψ,
    )
    return clamp(min(dt, p.dt_load), 1.0e-6, 1.0e9)
end

"""
True when the fixed point converged and the slip rate at most doubled.

A step chosen from a locked rate can converge on `V_cap` in one pass. Doubling
is slow enough for the weakening limit to track the event.
"""
function step_is_resolved(p, V_old, out; growth = 2.0)
    out.rel < p.tol || return false
    out.V < 0.5 * p.V_cap || return false
    return out.V <= max(V_old, 1.0e-30) * growth
end

"""Integrate the slider. Returns one named tuple per accepted step."""
function integrate_slider(p; nsteps = 20_000, max_time = 20 * 365.25 * 86400.0, dt_floor = 1.0e-3, max_events = 0)
    τ, ψ, V = p.τ, p.ψ, p.V
    t = 0.0
    hist = NamedTuple[]
    slipping = false
    events = 0
    for it in 1:nsteps
        t >= max_time && break
        dt = min(slider_dt(p, V, ψ), max_time - t)
        out = slider_step(p, τ, ψ, V, dt)
        cuts = 0
        # A locked timestep can jump the rate. Halve while the bracketed rate
        # more than doubles, but keep the step once dt is at the floor: that
        # jump is then the radiation-damped equilibrium, not a failed iteration.
        while !step_is_resolved(p, V, out) && dt > dt_floor && cuts < 24
            dt = max(dt * 0.5, dt_floor)
            out = slider_step(p, τ, ψ, V, dt)
            cuts += 1
        end
        isfinite(out.τ) && isfinite(out.ψ) && isfinite(out.V) ||
            error("slider went non-finite at it=$it t=$t")
        τ, ψ, V = out.τ, out.ψ, out.V
        t += dt
        push!(hist, (; it, t, dt, τ, ψ, V, passes = out.passes, rel = out.rel, cuts))
        if !slipping && V >= 1.0e-6
            slipping = true
        elseif slipping && V <= 1.0e-9
            slipping = false
            events += 1
            max_events > 0 && events >= max_events && break
        end
    end
    return hist
end

"""
Count locked → fast → locked passages.

`V_slip` opens an event, `V_lock` closes it. The drop is the stress at the
start of the event minus the minimum during it.
"""
function summarize_stickslip(hist; V_slip = 1.0e-6, V_lock = 1.0e-9)
    phase = :locked
    n = 0
    drops = Float64[]
    vmax = Float64[]
    τ_locked = isempty(hist) ? 0.0 : hist[1].τ
    τ_peak = τ_locked
    τ_min = τ_locked
    v_peak = 0.0
    for s in hist
        if phase === :locked
            if s.τ > τ_locked
                τ_locked = s.τ
            end
            if s.V >= V_slip
                phase = :slipping
                τ_peak = max(τ_locked, s.τ)
                τ_min = s.τ
                v_peak = s.V
            end
        else
            τ_peak = max(τ_peak, s.τ)
            τ_min = min(τ_min, s.τ)
            v_peak = max(v_peak, s.V)
            if s.V <= V_lock
                phase = :locked
                n += 1
                push!(drops, τ_peak - τ_min)
                push!(vmax, v_peak)
                τ_locked = s.τ
            end
        end
    end
    return (; n, drops, vmax)
end
