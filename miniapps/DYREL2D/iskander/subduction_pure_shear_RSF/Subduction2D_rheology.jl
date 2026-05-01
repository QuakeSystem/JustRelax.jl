using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rsf_params_lamem_simple_shear(di_min; V0_model = 4.0e-9)
    # Parameters aligned with the LaMEM RSF example shared earlier:
    # a_rsf=0.011, b_rsf=0.017, mu0_rsf=0.2, L_rsf=0.01, state_rsf_init=40, V0_rsf=4e-9.
    # `V0` / `p_shift` / solver knobs mirror `stress_kernels.jl` + `ratestate_0d_coupled` tests.
    return (
        active = (false, false, true, false),
        λ = (0.0, 0.0, 0.0, 0.0),
        μ0 = (0.2, 0.2, 0.2, 0.0),
        a = (0.011, 0.011, 0.005, 0.0),
        b = (0.017, 0.017, 0.007, 0.0),
        L = (0.01, 0.01, 0.01, 1.0),
        C = (0e0, 0.0, 0.0, 0.0),
        D = (di_min, di_min, di_min, di_min),
        Ω_init = (40.0, 40.0, 0.0, 40.0),
        V0 = (V0_model, V0_model, V0_model, V0_model),
        V0_model = V0_model,
        Vp_max = 1.0e19,
        p_shift = (0.0, 0.0, 0.0, 0.0),
        η_min = (1.0e1, 1.0e1, 1.0e1, 1.0e1),
        η_max = (1.0e23, 1.0e23, 1.0e23, 1.0e23),
        use_bisection = (true, true, true, false),
        maxit = (120, 120, 120, 8),
        rtol = (1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6),
    )
end


function init_rheology_simple_shear()

    ϕ_wet_olivine = asind(0.1)
    C_wet_olivine = 1.0e6
    η_reg = 1.0e20
    # Elasticity
    el = ConstantElasticity(; G = 3.0e10, ν = 0.49)
    asthenosphere = CompositeRheology(
        (
           # el,
            LinearViscous(; η = 1.0e18),
        )
    )

    lithosphere = CompositeRheology(
        (
            el,
            LinearViscous(; η = 1.0e23),
        )
    )
    subduction_lithosphere = CompositeRheology(
        (
            el,
            LinearViscous(; η = 1.0e23),
           #DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp = η_reg, Ψ = 0.0), # non-regularized plasticity
       
        )
    )
    rheologies = (;asthenosphere, lithosphere, subduction_lithosphere)
    return init_rheologies2(rheologies)
end


function init_rheologies2(rheologies)
    # common physical properties
    α = 2.4e-5 # 1 / K
    Cp = 750    # J / kg K

    el_bg = SetConstantElasticity(; G = 3.0e10, ν = 0.25)
    # Define rheolgy struct
    return rheology = (
            SetMaterialParams(;
            Name = "Asthenosphere",
            Phase = 1,
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.asthenosphere,
            Gravity = ConstantGravity(; g = 9.81),
            Elasticity = el_bg,
        ),
        SetMaterialParams(;
            Name = "Lithosphere",
            Phase = 2,
            Density = PT_Density(; ρ0 = 2.7e3, α = α, β = 0.0e0, T0 = 273 + 1000),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.lithosphere,
            Gravity = ConstantGravity(; g = 9.81),
            Elasticity = el_bg,
        ),

        SetMaterialParams(;
            Name = "Subduction Lithosphere",
            Phase = 3,
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = rheologies.subduction_lithosphere,
            Gravity = ConstantGravity(; g = 9.81),
            Elasticity = el_bg,
        ),

        SetMaterialParams(;
            Name = "StickyAir",
            Phase = 4,
            Density = ConstantDensity(; ρ = 100), # water density
            HeatCapacity = ConstantHeatCapacity(; Cp = 3.0e3),
            Conductivity = ConstantConductivity(; k = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e18),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )
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
            
            LinearViscous(; η = 1.0e23),
            # disl_wet_olivine,
            # diff_wet_olivine,
            # DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp = η_reg, Ψ = 0.0), # non-regularized plasticity
        )
    )
    return init_rheologies(lithosphere_rheology)
end



function init_rheologies(lithosphere_rheology)
    # common physical properties
    α = 2.4e-5 # 1 / K
    Cp = 750    # J / kg K
    gravity = 0.0
    # Define rheolgy struct
    return rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 3.2e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),)),
            Gravity = ConstantGravity(; g = gravity),
        ),
        # Name              = "Oceanic lithosphere",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 3.2e3, α = α, β = 0.0e0, T0 = 273 + 1474),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = lithosphere_rheology,
            Gravity = ConstantGravity(; g = gravity),
        ),
        # Name              = "oceanic crust",
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 3.2e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = lithosphere_rheology,
            Gravity = ConstantGravity(; g = gravity),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 4,
            Density = ConstantDensity(; ρ = 100), # water density
            HeatCapacity = ConstantHeatCapacity(; Cp = 3.0e3),
            Conductivity = ConstantConductivity(; k = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19),)),
            Gravity = ConstantGravity(; g = gravity),
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
