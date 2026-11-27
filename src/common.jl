using Adapt, MuladdMacro
import ForwardDiff

include("types/constructors/stokes.jl")
export StokesArrays, PTStokesCoeffs

include("types/constructors/heat_diffusion.jl")
export ThermalArrays, PTThermalCoeffs

include("types/constructors/weno.jl")
export WENO5

include("Utils.jl")
export @allocate,
    @add,
    @idx,
    @copy,
    @velocity,
    @velocity_old,
    @displacement,
    @strain,
    @plastic_strain,
    @stress,
    @tensor,
    @shear,
    @normal,
    @stress_center,
    @strain_center,
    @strain_increment,
    @tensor_center,
    @tensor_vertex,
    @shear_center,
    @qT,
    @qT2,
    @residuals,
    compute_dt,
    compute_lithostatic_pressure!,
    multi_copy!,
    take

include("types/displacement.jl")
export velocity2displacement!, displacement2velocity!

include("mask/constructors.jl")

include("boundaryconditions/BoundaryConditions.jl")
export AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions,
    flow_bcs!,
    thermal_bcs!,
    pureshear_bc!,
    simpleshear_bc!

include("MiniKernels.jl")

include("phases/phases.jl")
export fn_ratio

include("phases/PhaseRatios.jl")
export update_phase_ratios_2D!, update_phase_ratios_3D!

include("rheology/BuoyancyForces.jl")
export compute_ρg!, compute_ρ!

include("rheology/Viscosity.jl")
include("rheology/Viscosity_VS.jl")
export compute_viscosity!, compute_viscosity_εII!, compute_viscosity_τII!

import JustRelax: rsf_do_center, rsf_do_vertex

# Local allocator. `using ..JustRelax` already binds the parent struct, and an
# unqualified method would extend that constructor from both dimension modules.
function RateStateArrays end

export RateStateArrays, rsf_do_center, rsf_do_vertex

"""
    RateStateArrays(ni; loc=:both)
    RateStateArrays(backend, ni; loc=:both)

Allocate RSF center/vertex fields with this module's `@zeros` (Array or device).
GPU extensions override `RateStateArrays(::Type{CUDABackend}, ni)`.
"""
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
    return JustRelax.RateStateArrays{typeof(Ω), typeof(Ωv), eltype(Ω)}(
        Ω, Ω_old, Vp, τ_rsf, a_eff, b_eff, rsf_mask,
        Ωv, Ω_old_v, Vp_v, τ_rsf_v, a_eff_v, b_eff_v, rsf_mask_v,
        loc,
    )
end

RateStateArrays(backend, ni::NTuple{N, Integer}; kwargs...) where {N} =
    RateStateArrays(ni; kwargs...)

"""
    RateStateXY(ni; skip_vs=false)

Allocate xy-edge rate-and-state arrays and a `Vx` Dirichlet pair for a 3D grid
of `ni` cells. `skip_vs` drops `a ≥ b` edges from the stable timestep.
"""
RateStateXY(::Type{CPUBackend}, ni::NTuple{3, Integer}; kwargs...) = RateStateXY(ni; kwargs...)

function RateStateXY(ni::NTuple{3, Integer}; skip_vs::Bool = false)
    nx, ny, nz = ni
    Ω = @zeros(nx + 1, ny + 1, nz)
    vx_fixed = @zeros(nx + 1, ny + 2, nz + 2)
    return JustRelax.RateStateXY(
        Ω,
        @zeros(nx + 1, ny + 1, nz),
        @zeros(nx + 1, ny + 1, nz),
        @zeros(nx + 1, ny + 1, nz),
        @zeros(nx + 1, ny + 1, nz),
        @zeros(nx + 1, ny + 1, nz),
        @zeros(nx + 1, ny + 1, nz),
        @zeros(nx + 1, ny + 1, nz),
        vx_fixed,
        @zeros(nx + 1, ny + 2, nz + 2),
        skip_vs,
    )
end

include("rheology/Melting.jl")
export compute_melt_fraction!, compute_melt_fraction_derivative!

include("rheology/Solubility.jl")
export compute_dissolved_volatiles!

include("particles/subgrid_diffusion.jl")
export subgrid_characteristic_time!

include("Interpolations.jl")
export vertex2center!,
    center2vertex!, velocity2vertex!, velocity2center!, shear2center!

include("advection/weno5.jl")
export WENO_advection!

include("phases/topography_correction.jl")
export update_phases_given_markerchain!

# Stokes

include("rheology/GeoParams.jl")

include("rheology/StressUpdate.jl")
include("rheology/CapReturnMapping.jl")
export compute_yieldfunction_phase, compute_plastic_gradients_phase

include("stokes/StressKernels.jl")
export tensor_invariant!, accumulate_tensor!, accumulate_vol!

# GeoParams rate-and-state needs clamped_indices / av_clamped from StressKernels.
include("rheology/GeoParamsRateState.jl")
export init_geoparams_rate_state!,
    apply_geoparams_rsf_viscosity!,
    update_geoparams_rate_state!,
    compute_dt_geoparams_rsf,
    RateStateXY,
    seas_bp1_tau0,
    seas_bp1_omega,
    init_antiplane_fault!,
    apply_geoparams_rsf_dirichlet!

include("stokes/PrincipalStresses.jl")
export compute_principal_stresses, compute_principal_stresses!, PrincipalStress

include("stokes/PressureKernels.jl")
export rotate_stress_particles!

include("stokes/VelocityKernels.jl")

# variational Stokes
include("variational_stokes/mask.jl")
export RockRatio, update_rock_ratio!

include("variational_stokes/PressureKernels.jl")

include("variational_stokes/MiniKernels.jl")

include("variational_stokes/StressKernels.jl")

include("variational_stokes/VelocityKernels.jl")

include("stress_rotation/constructors.jl")
export StressParticles

include("stress_rotation/stress_rotation_particles.jl")
export rotate_stress!, stress2grid!

## DYREL solver

include("DYREL/constructors.jl")
include("DYREL/pressure_kernels.jl")
include("DYREL/stress_kernels.jl")
include("DYREL/velocity_kernels.jl")
include("DYREL/velocity_kernels_VS.jl")
include("DYREL/Gershgorin.jl")
include("DYREL/Gershgorin_VS.jl")

# thermal diffusion

include("thermal_diffusion/DiffusionPT.jl")
export PTThermalCoeffs, heatdiffusion_PT!, compute_shear_heating!

# Solver entry points forward their keywords to the backend-trait methods as a
# single `kwargs` NamedTuple. Callers may pass the options either as plain
# keywords or pre-bundled as `kwargs = (; ...)`; both normalize to the same
# NamedTuple here.
function flatten_solver_kwargs(kwargs)
    options = (; kwargs...)
    haskey(options, :kwargs) || return options
    bundle = options.kwargs
    bundle isa NamedTuple ||
        throw(ArgumentError("`kwargs` must be a NamedTuple, got $(typeof(bundle))"))
    return merge(bundle, Base.structdiff(options, (; kwargs = bundle)))
end
