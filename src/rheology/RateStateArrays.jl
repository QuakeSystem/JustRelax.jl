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

"""
    RateStateXY

GeoParams rate-and-state on the 3D `xy` shear edges (strike-slip on a fault whose
normal is `y` and whose slip is `x`). Array sizes match `τ.xy`: `(nx+1, ny+1, nz)`.

`η` is the creep viscosity the 3D stress kernel reads instead of the harmonic mean
of the cell viscosities. `vx_fixed` / `vx_value` are a Dirichlet mask on `Vx`
(plate flanks and the imposed deep slip). `skip_vs` keeps velocity-strengthening
edges (`a ≥ b`) out of the stable timestep.
"""
struct RateStateXY{A, V}
    Ω::A
    Ω_old::A
    Vp::A
    τ_rsf::A
    a_eff::A
    b_eff::A
    rsf_mask::A
    η::A
    vx_fixed::V
    vx_value::V
    skip_vs::Bool
end

Adapt.@adapt_structure RateStateArrays
Adapt.@adapt_structure RateStateXY

@inline rsf_do_center(loc::Symbol) = loc === :center || loc === :both
@inline rsf_do_vertex(loc::Symbol) = loc === :vertex || loc === :both
@inline rsf_do_center(rsf::RateStateArrays) = rsf_do_center(rsf.loc)
@inline rsf_do_vertex(rsf::RateStateArrays) = rsf_do_vertex(rsf.loc)
