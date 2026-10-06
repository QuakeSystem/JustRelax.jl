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
