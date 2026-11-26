"""
Maxwell viscoelastic media + fault buffer. RSF frictional softening is applied
inside `solve_DYREL!` via the `rsf=` hook (not GeoParams plasticity).
"""
function init_rheology_simple_shear(; η = 5.0e26, G = 3.0e10, ν = 0.25)
    η0 = LinearViscous(; η = η)
    el = ConstantElasticity(; G = G, ν = ν)
    media_rheology = CompositeRheology((η0, el))
    vel_weak_rheology = CompositeRheology((η0, el))
    return init_rheologies((; media_rheology, vel_weak_rheology); G = G, ν = ν)
end

"""
Series composite used by the GeoParams rate-and-state path: linear dashpot, explicit
`GeoParams.RateStateFriction`, and elasticity. The fault phase stores the background
`b`; a spatial `b(x)` is applied per vertex through `b_eff`.
"""
function init_rheology_rate_state(;
        η = 5.0e26,
        G = 3.0e10,
        ν = 0.25,
        V₀ = 4.0e-9,
        a = 0.011,
        b_media = 0.017,
        b_fault = 0.001,
        μ₀ = 0.2,
        L = 0.01,
        Wf = 500.0,   # LaMEM fault width [m]; stored on the law as D, Vp = 2 Wf ε
        D = Wf,
        λ = 0.0,
        C = 0.0,
    )
    η0 = LinearViscous(; η = η)
    el = ConstantElasticity(; G = G, ν = ν)
    rsf_media = GeoParams.RateStateFriction(λ, μ₀, V₀, a, b_media, L, C, D)
    rsf_fault = GeoParams.RateStateFriction(λ, μ₀, V₀, a, b_fault, L, C, D)
    media_rheology = CompositeRheology((η0, rsf_media, el))
    vel_weak_rheology = CompositeRheology((η0, rsf_fault, el))
    return init_rheologies((; media_rheology, vel_weak_rheology); G = G, ν = ν)
end

function init_rheologies(rheologies; G = 3.0e10, ν = 0.25)
    Cp = 750.0
    el_bg = ConstantElasticity(; G = G, ν = ν)
    return (
        SetMaterialParams(;
            Name = "Media",
            Phase = 1,
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.media_rheology,
            Gravity = ConstantGravity(; g = 0.0),
            Elasticity = el_bg,
        ),
        SetMaterialParams(;
            Name = "Velocity weakening",
            Phase = 2,
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.vel_weak_rheology,
            Gravity = ConstantGravity(; g = 0.0),
            Elasticity = el_bg,
        ),
    )
end
