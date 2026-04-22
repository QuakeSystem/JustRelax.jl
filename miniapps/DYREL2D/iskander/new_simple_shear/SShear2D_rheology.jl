using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheology_nonNewtonian()
    #dislocation laws
    disl_wet_olivine = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)

    el = ConstantElasticity(; G = 40.0e9)

    lithosphere_rheology = CompositeRheology((el, disl_wet_olivine, diff_wet_olivine))
    rheologies = (;lithosphere_rheology)
    return init_rheologies(rheologies)
end

function init_rheology_nonNewtonian_plastic()
    #dislocation laws
    disl_wet_olivine = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
    # plasticity
    ϕ_wet_olivine = asind(0.1)
    C_wet_olivine = 1.0e6
    η_reg = 1.0e20
    el = ConstantElasticity(; G = 40.0e9, ν = 0.25)
    lithosphere_rheology = CompositeRheology(
        (
            el,
            disl_wet_olivine,
            diff_wet_olivine,
            #DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp = η_reg, Ψ = 0.0), # non-regularized plasticity
        )
    )
    rheologies = (;lithosphere_rheology)
    return init_rheologies(rheologies)
end

function init_rheology_simple_shear()
    # plasticity
    ϕ_wet_olivine = 37 #asind(0.1)
    C_wet_olivine = 1.0e7
    η_reg = 1.0e20
    # Elasticity
    el = ConstantElasticity(; G = 3.0e10, ν = 0.49)
    media_rheology = CompositeRheology(
        (
            el,
            LinearViscous(; η = 1.0e22),
            #DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp = η_reg, Ψ = 0.0), # non-regularized plasticity
        )
    )
    ϕ_wet_olivine = 5 #asind(0.1)
    C_wet_olivine = 1.0e6
    vel_weak_rheology = CompositeRheology(
        (
            el,
            #DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp = η_reg, Ψ = 0.0), # non-regularized plasticity
        )
    )
    vel_strength_rheology =vel_weak_rheology;
    rheologies = (;media_rheology, vel_weak_rheology, vel_strength_rheology)
    return init_rheologies(rheologies)
end

function init_rsf_params_simple_shear(di_min)
    # Phase-dependent RSF-like parameters used by the DYREL stress update hook.
    # Phase ordering follows init_rheologies:
    # 1: media, 2: velocity weakening, 3: velocity strengthening, 4: sticky air.
    return (
        active = (false, false, false, false),              # phase-wise RSF activation
        mu_d = (0.5, 0.15, 0.15, 0.0),          # dynamic friction coefficient
        mu_s = (0.7, 0.5, 0.3, 0.0),          # static friction coefficient
        sigma_c = (1.0e7, 0.0e0, 0e0, 0.0),    # compressive strength [Pa]
        Vc = (1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8),   # characteristic slip velocity [m/s]
        D = (di_min, di_min, di_min, di_min),    # characteristic length scale [m]
        maxit = (8, 8, 8, 8),                    # fixed-point iterations
        rtol = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3), # local convergence tolerance
    )
    # return (
    #     active = (true, true, true, false),              # phase-wise RSF activation
    #     mu_d = (0.5, 0.5, 0.5, 0.0),          # dynamic friction coefficient
    #     mu_s = (0.7, 0.7, 0.7, 0.0),          # static friction coefficient
    #     sigma_c = (1.0e7, 1.0e7, 1.0e7, 0.0),    # compressive strength [Pa]
    #     Vc = (1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8),   # characteristic slip velocity [m/s]
    #     D = (di_min, di_min, di_min, di_min),    # characteristic length scale [m]
    #     maxit = (8, 8, 8, 8),                    # fixed-point iterations
    #     rtol = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3), # local convergence tolerance
    # )
end

function init_rheology_linear()
    el = ConstantElasticity(; G = 40.0e9, ν = 0.45)
    # lithosphere_rheology = CompositeRheology( (LinearViscous(; η=1e23), ))
    lithosphere_rheology = CompositeRheology((LinearViscous(; η = 1.0e23), el))
    rheologies = (;lithosphere_rheology)
    return init_rheologies(rheologies)
end

function init_rheologies(rheologies)
    # common physical properties
    α = 2.4e-5 # 1 / K
    Cp = 750    # J / kg K

    el_bg = SetConstantElasticity(; G = 3.0e10, ν = 0.25)
    # Define rheolgy struct
    return rheology = (
        # Name = "Media",
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
        # Name              = "Velocity weakening",
        SetMaterialParams(;
            Name = "Velocity weakening",
            Phase = 2,
            Density = PT_Density(; ρ0 = 2.7e3, α = α, β = 0.0e0, T0 = 273 + 1000),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.vel_weak_rheology,
            # CompositeRheology = rheologies.media_rheology,
            Gravity = ConstantGravity(; g = 0.0),
            Elasticity = el_bg,
        ),
        # Name              = "Velocity strengthening",
        SetMaterialParams(;
            Name = "Velocity strengthening",
            Phase = 3,
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.vel_strength_rheology,
            # CompositeRheology = rheologies.media_rheology,
            Gravity = ConstantGravity(; g = 0.0),
            Elasticity = el_bg,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Name = "StickyAir",
            Phase = 4,
            Density = ConstantDensity(; ρ = 100), # water density
            HeatCapacity = ConstantHeatCapacity(; Cp = 3.0e3),
            Conductivity = ConstantConductivity(; k = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19),)),
            Gravity = ConstantGravity(; g = 0.0),
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N, T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
