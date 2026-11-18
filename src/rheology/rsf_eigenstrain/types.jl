"""
    EigenstrainRSF

Fault-row state for the elastic eigenstrain rate-and-state split.

Arrays are vertex-sized (`ni .+ 1`), same layout as `τ.xy`. Only nodes with
`mask > 0.5` are the fault. `Wf` is the LaMEM fault width: the thickness of the
zone, usually 25–500 m, in the eigenstrain `V / (2 Wf)`.

SEAS BP1-FD uses the symbol `Wf` for a different length, the down-dip height of
the rate-and-state section (40 km in that benchmark). That length is `Wf_height`,
not this field.

`law` is a `RateStateFriction` prototype (`μ₀`, `V₀`, `L`, `λ = 0`). Nodal `a` and `b`
replace the prototype values. `σ` is the effective normal stress used in the slip law
(not the dynamic pressure).
"""
struct EigenstrainRSF{A, T, L}
    mask::A
    ψ::A
    ψ_old::A
    V_lag::A
    V::A
    a::A
    b::A
    Wf::T
    G::T
    σ::T
    V_cap::T
    tol::T
    max_pass::Int
    law::L
end

Adapt.@adapt_structure EigenstrainRSF
