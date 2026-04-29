# Shared 0D march: RheologyCalculator `solve` drives adaptive `dt`; JustRelax `_compute_local_stress`
# sees the same Ω_old and simple-shear τ memory as in `RheologyCalculator.jl/examples/RateState_compare_JustRelax.jl`.
# Loaded by `test_rsf_rc_0d_snapshots.jl` (not a test file on its own).

using JustRelax
using RheologyCalculator
import RheologyCalculator: compute_stress_elastic

function compute_dt_ratestate(rs::RateStateFriction, re::AbstractElasticity, P, Vp, Ω; dt_min = 1e-4, dt_max = 1e6, f_courant = 1e-3)
    dt_c = dt_courant(Vp, rs.D; f = f_courant)
    dt_h = dt_healing(rs, Ω)
    θmax = max_state_change(rs, re, P)
    dt_w = dt_weakening(rs, θmax, Vp)
    dt = min(dt_c, dt_h, dt_w)
    return clamp(dt, dt_min, dt_max)
end

dt_healing(r::RateStateFriction, Ω) = 0.2 * r.L / (r.V₀ * exp(-Ω))
dt_weakening(r::RateStateFriction, θmax, Vp) = θmax * r.L / Vp
dt_courant(Vmax, Δ; f = 1e-3) = f * Δ / Vmax

function max_state_change(rs::RateStateFriction, re::AbstractElasticity, P)
    Peff = P * (1 - rs.λ)
    faultwidth = rs.D
    if isa(re, IncompressibleElasticity)
        G = re.G
        ν = 0.5
    else
        G = re.G
        ν = re.ν
    end
    k = 2 / π * (G / (1 - ν)) / faultwidth
    xi = 0.25 * (k * rs.L / (rs.a * Peff) - (rs.b - rs.a) / rs.a)^2 - k * rs.L / (rs.a * Peff)
    if xi > 0
        θmax = rs.a * Peff / (k * rs.L - (rs.b - rs.a) * Peff)
    elseif xi < 0
        θmax = 1 - (rs.b - rs.a) * Peff / (k * rs.L)
    else
        error("xi=0 in time step calculation")
    end
    θmax = clamp(θmax, 0.1, 0.2)
    return θmax
end

function stress_time_coupled(
    c,
    vars,
    x,
    xnorm,
    others,
    hyp::RateStateFriction,
    elastic::IncompressibleElasticity;
    ntime = 1000,
    dt0 = 1e6,
    η_bg = 1.0e40,
)
    G = elastic.G
    Kb = 50.0e9
    P = others.P[1]
    ε_drv = vars.ε
    eij = (ε_drv, -ε_drv, 0.0)

    rsf = (
        active = (true,),
        a = (hyp.a,),
        b = (hyp.b,),
        μ0 = (hyp.μ₀,),
        V0 = (hyp.V₀,),
        L = (hyp.L,),
        D = (hyp.D,),
        λ = (hyp.λ,),
        C = (hyp.C,),
        η_min = (1.0e18,),
        η_max = (1.0e24,),
        Vp_max = (1.0e19,),
        p_shift = (0.0,),
        use_bisection = (false,),
        maxit = (12,),
        rtol = (1.0e-6,),
    )

    τ_rc = zeros(ntime)
    τ_jr = zeros(ntime)
    τy_rc = zeros(ntime)
    τy_jr = zeros(ntime)
    Ω_rc = zeros(ntime)
    Ω_jr = zeros(ntime)
    V_rc = zeros(ntime)
    V_jr = zeros(ntime)
    t_v = zeros(ntime)

    τ_e = others.τ0
    Ωc = 0.0
    P_e = others.P
    dt = dt0
    t = 0.0
    xlocal = x

    τ_rc[1] = τ_e[1]
    τ_jr[1] = τ_e[1]
    τy_rc[1] = P * (hyp.μ₀ + hyp.b * 0)
    τy_jr[1] = τy_rc[1]

    for i in 2:ntime
        Ωc_in = Ωc
        others_i = (; dt = dt, τ0 = τ_e, Ω_old = Ωc_in, P0 = P_e, P = (50e6,))
        xlocal = solve(c, xlocal, vars, others_i, verbose = false, xnorm = xnorm)
        t += dt
        ε_ratestate = compute_strain_rate(c[1]; τ = xlocal[1], Ω_old = others_i.Ω_old, P = others_i.P[1], dt = others_i.dt)
        τ_e = compute_stress_elastic(c, xlocal, others_i)
        Vp_rc = 2.0 * c[1].D * ε_ratestate
        Ωc = update_Ω(c[1]; ε = ε_ratestate, others_i...)
        τ_rc[i] = xlocal[1]
        τy_rc[i] = others_i.P[1] * (c[1].μ₀ + c[1].b * Ωc)
        Ω_rc[i] = Ωc
        V_rc[i] = Vp_rc
        t_v[i] = t

        τ_hist = (τ_rc[i - 1], -τ_rc[i - 1], 0.0)
        _, _, _, _, _, _, τII, _, _, _, Ωjn, Vp_j = JustRelax.JustRelax2D._compute_local_stress(
            eij,
            τ_hist,
            η_bg,
            P,
            G,
            Kb,
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
            Ωc_in,
        )
        τ_jr[i] = τII
        τy_jr[i] = P * (hyp.μ₀ + hyp.b * Ωjn)
        Ω_jr[i] = Ωjn
        V_jr[i] = Vp_j

        dt = compute_dt_ratestate(c[1], c[2], others_i.P[1], Vp_rc, Ωc; dt_min = 1e-4, dt_max = 1e6)
    end
    return t_v, τ_rc, Ω_rc, V_rc, τy_rc, τ_jr, Ω_jr, V_jr, τy_jr
end

function default_ratestate_series_setup()
    a = 0.011
    b = 0.015
    L = 0.0047
    μ₀ = 0.5
    V₀ = 4e-9
    λ = 0
    C = 0
    D = 500
    hyp = RateStateFriction(λ, μ₀, V₀, a, b, L, C, D)
    elastic = IncompressibleElasticity(20e9)
    c = SeriesModel(hyp, elastic)
    platerate = 4e-9
    vars = (; ε = platerate / D / 2, θ = 1.0e-20)
    args = (; τ = 2.0e1, P = 50.0e6)
    others = (; dt = 1.1e-2, τ0 = (2e7,), P0 = (0.0,), P = (50e6,))
    x = initial_guess_x(c, vars, args, others)
    char_τ = 1e6
    char_ε = vars.ε + vars.θ
    xnorm = normalisation_x(c, char_τ, char_ε)
    return hyp, elastic, c, x, xnorm, vars, others
end
