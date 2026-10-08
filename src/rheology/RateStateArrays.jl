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

@inline rsf_do_center(loc::Symbol) = loc === :center || loc === :both
@inline rsf_do_vertex(loc::Symbol) = loc === :vertex || loc === :both
@inline rsf_do_center(rsf::RateStateArrays) = rsf_do_center(rsf.loc)
@inline rsf_do_vertex(rsf::RateStateArrays) = rsf_do_vertex(rsf.loc)
