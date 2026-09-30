function init_rheology_simple_shear(; η = 1.0e23)
    # High η → Maxwell buffer is quasi-elastic: η_ve ≈ G*dt when η ≫ G*dt
    # Herrendörfer / LaMEM RSF shear test uses η = 5e26
    η0 = LinearViscous(; η = η)
    el = ConstantElasticity(; G = 3.0e10, ν = 0.25)
    media_rheology = CompositeRheology((η0, el))
    # Fault starts with the same high η; RSF supplies frictional η_eff via APT hook
    vel_weak_rheology = CompositeRheology((η0, el))

    rheologies = (; media_rheology, vel_weak_rheology)
    return init_rheologies(rheologies)
end


function init_rheologies(rheologies)
    # common physical properties
    Cp = 750    # J / kg K

    el_bg = SetConstantElasticity(; G = 3.0e10, ν = 0.25)
    # Define rheolgy struct
    return rheology = (
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
        )
    )
end


function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N, T}

    # phase_grid / xvi are on vertices (ni.+1); phases are on cells (ni)
    nvx, nvy = length(xvi[1]), length(xvi[2])

    for ip in cellaxes(phases)
        # quick escape (JustPIC index is Bool: false/true)
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest vertex
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            # need lower+upper bounds: cell corners include I+1 up to nv
            !(1 ≤ ii ≤ nvx && 1 ≤ jj ≤ nvy) && continue
            !(1 ≤ ii ≤ size(phase_grid, 1) && 1 ≤ jj ≤ size(phase_grid, 2)) && continue

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
