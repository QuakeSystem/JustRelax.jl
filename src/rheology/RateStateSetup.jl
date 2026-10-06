# RSF setup validation, LaMEM-style spatial a/b profiles, and controller build.

"""
    get_rsf_profile(x, x_knots, val_knots)

Piecewise-linear interpolation of an RSF parameter along an axis (LaMEM `getRsfProfile`).
Clamps to endpoint values outside the knot range.
"""
@inline function get_rsf_profile(x::Real, x_knots, val_knots)
    n = length(x_knots)
    @assert n == length(val_knots) && n ≥ 1
    if x ≤ x_knots[1]
        return float(val_knots[1])
    elseif x ≥ x_knots[n]
        return float(val_knots[n])
    end
    @inbounds for i in 1:(n - 1)
        x0, x1 = x_knots[i], x_knots[i + 1]
        if x0 ≤ x ≤ x1
            t = (x - x0) / (x1 - x0)
            return float(val_knots[i]) * (1 - t) + float(val_knots[i + 1]) * t
        end
    end
    return float(val_knots[end])
end

const _RSF_REQUIRED = (:b_rsf, :mu0_rsf, :D_rs, :state_rsf_init)

"""
    phase_has_rsf(phase_nt) -> Bool

A phase uses RSF iff `a_rsf` is set (and not `nothing`).
"""
@inline function phase_has_rsf(phase_nt)
    return hasproperty(phase_nt, :a_rsf) && !isnothing(getproperty(phase_nt, :a_rsf))
end

"""
    validate_rsf_phase!(phase_nt, phase_label)

If `a_rsf` is present, assert the other RSF fields exist.
"""
function validate_rsf_phase!(phase_nt, phase_label)
    phase_has_rsf(phase_nt) || return nothing
    missing = Symbol[]
    for k in _RSF_REQUIRED
        if !hasproperty(phase_nt, k) || isnothing(getproperty(phase_nt, k))
            push!(missing, k)
        end
    end
    isempty(missing) || error(
        "RSF phase $(phase_label): `a_rsf` is set but missing required fields: $(join(missing, ", "))"
    )
    return nothing
end

"""
Per-phase RSF constitutive parameters (constants; spatial profiles live on grid fields).

`Wf` is the LaMEM fault-zone width (`Vp = 2 Wf εII`).
"""
struct PhaseRSFParams{T}
    enabled::Bool
    λ::T
    μ₀::T
    V₀::T
    a::T
    b::T
    L::T      # D_rs
    C::T
    Wf::T     # fault width
    state_init::T
end

Adapt.@adapt_structure PhaseRSFParams

function PhaseRSFParams(::Type{T}, phase_nt; V0_global, Wf_default) where {T}
    if !phase_has_rsf(phase_nt)
        z = zero(T)
        return PhaseRSFParams{T}(false, z, z, z, z, z, z, z, z, z)
    end
    λ = T(get(phase_nt, :λ, 0))
    μ₀ = T(phase_nt.mu0_rsf)
    V₀ = T(get(phase_nt, :V0, V0_global))
    a = T(phase_nt.a_rsf)
    b = T(phase_nt.b_rsf)
    L = T(phase_nt.D_rs)
    C = T(get(phase_nt, :C, 0))
    # LaMEM name Wf; accept legacy `D` as alias
    Wf_raw = if hasproperty(phase_nt, :Wf) && !isnothing(phase_nt.Wf)
        phase_nt.Wf
    elseif hasproperty(phase_nt, :D) && !isnothing(phase_nt.D)
        phase_nt.D
    else
        nothing
    end
    Wf = isnothing(Wf_raw) ? T(Wf_default) : T(Wf_raw)
    state_init = T(phase_nt.state_rsf_init)
    return PhaseRSFParams{T}(true, λ, μ₀, V₀, a, b, L, C, Wf, state_init)
end

@inline function as_rate_state_friction(p::PhaseRSFParams)
    return RateStateFriction(p.λ, p.μ₀, p.V₀, p.a, p.b, p.L, p.C, p.Wf)
end

"""
    RateStateController

Holds validated RSF setup: per-phase params, dt clamps, location flag.
Built from a miniapp `RSF` NamedTuple via [`build_rate_state_controller`](@ref).
"""
struct RateStateController{P, T}
    enabled::Bool
    affect_stokes::Bool  # if false: diagnostics only (Vp/Ω); no η fold into Stokes
    loc::Symbol          # :center, :vertex, or :both (LaMEM cell+edge)
    V0::T
    dt_min::T
    dt_max::T
    dt_rsf_switch::T     # use RSF dt if dt_rsf < this; else CFL
    G::T                 # shear modulus for Lapusta dt_w (LaMEM)
    ν::T                 # Poisson ratio for Lapusta dt_w
    phases::P            # NTuple{N, PhaseRSFParams{T}}
    # optional profile knots stored as NamedTuples per phase (host-side only)
    a_profiles::Any
    b_profiles::Any
end

function _get_phase_nt(rsf_nt, iphase::Int)
    key = Symbol("Phase", iphase)
    return hasproperty(rsf_nt, key) ? getproperty(rsf_nt, key) : nothing
end

function _profile_spec(phase_nt, prefix::Symbol)
    isnothing(phase_nt) && return nothing
    val_key = Symbol(prefix, :_rsf_val)
    x_key = Symbol(prefix, :_rsf_x)
    y_key = Symbol(prefix, :_rsf_y)
    if hasproperty(phase_nt, val_key) && hasproperty(phase_nt, x_key)
        return (; axis = :x, knots = getproperty(phase_nt, x_key), vals = getproperty(phase_nt, val_key))
    elseif hasproperty(phase_nt, val_key) && hasproperty(phase_nt, y_key)
        return (; axis = :y, knots = getproperty(phase_nt, y_key), vals = getproperty(phase_nt, val_key))
    end
    return nothing
end

"""
    build_rate_state_controller(rsf_nt; nphases, di, T=Float64)

Validate `RSF` NamedTuple, auto-enable phases with `a_rsf`, return [`RateStateController`](@ref).
`di` is used as default fault width `Wf = min(di)` when a phase omits `Wf`.
"""
function build_rate_state_controller(
        rsf_nt;
        nphases::Integer,
        di,
        T::Type = Float64,
    )
    enabled = Bool(get(rsf_nt, :enabled, true))
    affect_stokes = Bool(get(rsf_nt, :affect_stokes, true))
    loc = get(rsf_nt, :loc, :both)
    loc in (:center, :vertex, :both) ||
        error("RSF.loc must be :center, :vertex, or :both, got $loc")
    V0 = T(get(rsf_nt, :V0, 1.0e-9))
    dt_min = T(get(rsf_nt, :dt_min, 1.0e-2))
    dt_max = T(get(rsf_nt, :dt_max, 1.0e7))
    dt_rsf_switch = T(get(rsf_nt, :dt_rsf_switch, 1.0e9))
    G = T(get(rsf_nt, :G, 3.0e10))
    ν = T(get(rsf_nt, :ν, 0.25))
    Wf_default = T(minimum(di))

    phases = ntuple(nphases) do ip
        pnt = _get_phase_nt(rsf_nt, ip)
        if isnothing(pnt)
            z = zero(T)
            return PhaseRSFParams{T}(false, z, z, z, z, z, z, z, z, z)
        end
        validate_rsf_phase!(pnt, "Phase$ip")
        return PhaseRSFParams(T, pnt; V0_global = V0, Wf_default = Wf_default)
    end

    a_profiles = ntuple(ip -> _profile_spec(_get_phase_nt(rsf_nt, ip), :a), nphases)
    b_profiles = ntuple(ip -> _profile_spec(_get_phase_nt(rsf_nt, ip), :b), nphases)

    any_on = any(p -> p.enabled, phases)
    enabled = enabled && any_on

    return RateStateController(
        enabled, affect_stokes, loc, V0, dt_min, dt_max, dt_rsf_switch, G, ν,
        phases, a_profiles, b_profiles,
    )
end

"""
    evaluate_rsf_ab(ctrl, iphase, x, y) -> (a, b)

Constant or piecewise-linear `a`/`b` for phase `iphase` at coordinates `(x, y)`.
"""
@inline function evaluate_rsf_ab(ctrl::RateStateController, iphase::Integer, x, y)
    p = ctrl.phases[iphase]
    a = p.a
    b = p.b
    aprof = ctrl.a_profiles[iphase]
    bprof = ctrl.b_profiles[iphase]
    if !isnothing(aprof)
        ξ = aprof.axis === :x ? x : y
        a = typeof(a)(get_rsf_profile(ξ, aprof.knots, aprof.vals))
    end
    if !isnothing(bprof)
        ξ = bprof.axis === :x ? x : y
        b = typeof(b)(get_rsf_profile(ξ, bprof.knots, bprof.vals))
    end
    return a, b
end
