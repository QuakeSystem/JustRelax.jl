# Cell-centered (default) rate-and-state field arrays.

"""
    RateStateArrays

RSF state / diagnostic fields on the staggered grid (`loc = :center` by default).

- `Ω`, `Ω_old` — logarithmic state
- `Vp` — slip-rate diagnostic
- `a_eff`, `b_eff` — local constitutive `a`/`b` (constants or profiles)
- `rsf_mask` — 1 where the dominant phase has RSF enabled
"""
struct RateStateArrays{A, T}
    Ω::A
    Ω_old::A
    Vp::A
    a_eff::A
    b_eff::A
    rsf_mask::A
    loc::Symbol
end

Adapt.@adapt_structure RateStateArrays

function RateStateArrays(ni::NTuple{N, Integer}; loc::Symbol = :center) where {N}
    loc in (:center, :vertex) || error("RateStateArrays loc must be :center or :vertex")
    dims = loc === :center ? ni : ni .+ 1
    Ω = @zeros(dims...)
    Ω_old = @zeros(dims...)
    Vp = @zeros(dims...)
    a_eff = @zeros(dims...)
    b_eff = @zeros(dims...)
    rsf_mask = @zeros(dims...)
    return RateStateArrays{typeof(Ω), eltype(Ω)}(Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, loc)
end

RateStateArrays(backend, ni::NTuple{N, Integer}; kwargs...) where {N} = RateStateArrays(ni; kwargs...)
