# Rate-and-state field arrays on the staggered grid (LaMEM cell + edge).

"""
    RateStateArrays

RSF state / diagnostic fields at **cell centers** and **vertices** (JR shear / LaMEM
XY-edge nodes). Matches LaMEM `svCell` + `svXYEdge` each carrying their own `svDev`.

Center fields (`Ω`, `Vp`, …) are size `ni`; vertex fields (`Ωv`, `Vp_v`, …) are `ni .+ 1`.

`loc` selects which locations are updated (`:center`, `:vertex`, or `:both`; default `:both`).
"""
struct RateStateArrays{A, Av, T}
    # centers
    Ω::A
    Ω_old::A
    Vp::A
    τ_rsf::A
    a_eff::A
    b_eff::A
    rsf_mask::A
    # vertices (shear nodes)
    Ωv::Av
    Ω_old_v::Av
    Vp_v::Av
    τ_rsf_v::Av
    a_eff_v::Av
    b_eff_v::Av
    rsf_mask_v::Av
    loc::Symbol
end

Adapt.@adapt_structure RateStateArrays

function RateStateArrays(ni::NTuple{N, Integer}; loc::Symbol = :both) where {N}
    loc in (:center, :vertex, :both) ||
        error("RateStateArrays loc must be :center, :vertex, or :both, got $loc")
    Ω = @zeros(ni...)
    Ω_old = @zeros(ni...)
    Vp = @zeros(ni...)
    τ_rsf = @zeros(ni...)
    a_eff = @zeros(ni...)
    b_eff = @zeros(ni...)
    rsf_mask = @zeros(ni...)
    nv = ni .+ 1
    Ωv = @zeros(nv...)
    Ω_old_v = @zeros(nv...)
    Vp_v = @zeros(nv...)
    τ_rsf_v = @zeros(nv...)
    a_eff_v = @zeros(nv...)
    b_eff_v = @zeros(nv...)
    rsf_mask_v = @zeros(nv...)
    return RateStateArrays{typeof(Ω), typeof(Ωv), eltype(Ω)}(
        Ω, Ω_old, Vp, τ_rsf, a_eff, b_eff, rsf_mask,
        Ωv, Ω_old_v, Vp_v, τ_rsf_v, a_eff_v, b_eff_v, rsf_mask_v,
        loc,
    )
end

RateStateArrays(backend, ni::NTuple{N, Integer}; kwargs...) where {N} =
    RateStateArrays(ni; kwargs...)

@inline rsf_do_center(loc::Symbol) = loc === :center || loc === :both
@inline rsf_do_vertex(loc::Symbol) = loc === :vertex || loc === :both
@inline rsf_do_center(rsf::RateStateArrays) = rsf_do_center(rsf.loc)
@inline rsf_do_vertex(rsf::RateStateArrays) = rsf_do_vertex(rsf.loc)
