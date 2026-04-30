# Load script dependencies
using GeoParams#, CairoMakie
using Printf
# using Infiltrator
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
    const backend_JR = CUDABackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
    const backend_JR = CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# cd(@__DIR__)
# Load file with all the rheology configurations
include("SShear2D_setup.jl")
include("SShear2D_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

function wrap_particles_x!(particles, xvi)
    xmin = minimum(xvi[1])
    xmax = maximum(xvi[1])
    lx = xmax - xmin
    ppx = particles.coords[1].data
    idx = particles.index.data

    @parallel_indices (I) function _wrap_particles_x_kernel!(ppx, idx, xmin, xmax, lx)
        @inbounds if idx[I] > 0
            x = ppx[I]
            if x < xmin
                ppx[I] = x + lx
            elseif x >= xmax
                ppx[I] = x - lx
            end
        end
        return nothing
    end

    @parallel (@idx length(idx)) _wrap_particles_x_kernel!(ppx, idx, xmin, xmax, lx)
    return nothing
end

function enforce_periodic_phase_ratios_x!(phase_ratios)
    @parallel_indices (j) function _copy_periodic_vertex_x!(vertex)
        vertex[size(vertex, 1), j] = vertex[1, j]
        return nothing
    end
    @parallel (@idx (size(phase_ratios.vertex, 2))) _copy_periodic_vertex_x!(phase_ratios.vertex)
    return nothing
end

function residuals_to_center(Rx::AbstractMatrix, Ry::AbstractMatrix, RP::AbstractMatrix)
    nx, ny = size(RP)
    Rx_c = zeros(eltype(RP), nx, ny)
    Ry_c = zeros(eltype(RP), nx, ny)

    @inbounds for j in 1:ny
        for i in 1:nx
            il = max(i - 1, 1)
            ir = min(i, size(Rx, 1))
            Rx_c[i, j] = 0.5 * (Rx[il, j] + Rx[ir, j])
        end
    end

    @inbounds for j in 1:ny
        jb = max(j - 1, 1)
        jt = min(j, size(Ry, 2))
        for i in 1:nx
            Ry_c[i, j] = 0.5 * (Ry[i, jb] + Ry[i, jt])
        end
    end

    Rmag_c = @. sqrt(Rx_c^2 + Ry_c^2 + RP^2)
    return Rx_c, Ry_c, Rmag_c
end

@inline _rsf_pick(v::Number, ::Int) = v
@inline _rsf_pick(v::NTuple{N, <:Number}, phase::Int) where {N} = v[phase]
@inline _rsf_pick(v::AbstractVector{<:Number}, phase::Int) = v[phase]

@parallel_indices (I...) function _init_rsf_state_center!(Ω, Ω_new, Vp, phase_center, rsf_params)
    ratios = phase_center[I...]
    phase = argmax(ratios)
    Ω0 = _rsf_pick(rsf_params.Ω_init, phase)
    Ω[I...] = Ω0
    Ω_new[I...] = Ω0
    Vp[I...] = 0.0
    return nothing
end

@parallel_indices (I...) function _init_rsf_state_vertex!(Ω, Ω_new, Vp, phase_vertex, rsf_params)
    ratios = phase_vertex[I...]
    phase = argmax(ratios)
    Ω0 = _rsf_pick(rsf_params.Ω_init, phase)
    Ω[I...] = Ω0
    Ω_new[I...] = Ω0
    Vp[I...] = 0.0
    return nothing
end

function init_rsf_state_fields(ni, phase_ratios, rsf_params)
    Ω_center = @zeros(ni...)
    Ω_center_new = similar(Ω_center)
    Vp_center = @zeros(ni...)
    Ω_vertex = @zeros(ni .+ 1...)
    Ω_vertex_new = similar(Ω_vertex)
    Vp_vertex = @zeros(ni .+ 1...)
    @parallel (@idx ni) _init_rsf_state_center!(Ω_center, Ω_center_new, Vp_center, phase_ratios.center, rsf_params)
    @parallel (@idx ni .+ 1) _init_rsf_state_vertex!(Ω_vertex, Ω_vertex_new, Vp_vertex, phase_ratios.vertex, rsf_params)
    return (; Ω_center, Ω_center_new, Vp_center, Ω_vertex, Ω_vertex_new, Vp_vertex)
end

@inline dt_healing_rsf(L, V0, Ω) = 0.2 * L / max(V0 * exp(-Ω), eps(Float64))
@inline dt_weakening_rsf(L, θmax, Vp) = θmax * L / max(Vp, eps(Float64))
@inline dt_courant_rsf(Vp, Δ; f = 1.0e-3) = f * Δ / max(Vp, eps(Float64))

function max_state_change_rsf(a, b, L, D, λ, P, G, ν)
    Peff = max(P * (1.0 - λ), eps(Float64))
    k = 2.0 / π * (G / (1.0 - ν)) / max(D, eps(Float64))
    xi = 0.25 * (k * L / (a * Peff) - (b - a) / a)^2 - k * L / (a * Peff)
    θmax = if xi > 0.0
        a * Peff / max(k * L - (b - a) * Peff, eps(Float64))
    else
        1.0 - (b - a) * Peff / max(k * L, eps(Float64))
    end
    return clamp(θmax, 0.1, 0.2)
end

function compute_dt_ratestate_global(
    rsf_params,
    rsf_state,
    P;
    dt_min = 1.0e-4,
    dt_max = 1.0e8,
    f_courant = 1.0e-3,
    # Match `init_rheology_simple_shear` / `SetConstantElasticity` shear modulus (not 40 GPa).
    G = 3.0e10,
    ν = 0.49,
)
    # Use velocity-weakening layer (phase 2) as conservative RSF timestep controller.
    phase = 2
    a = _rsf_pick(rsf_params.a, phase)
    b = _rsf_pick(rsf_params.b, phase)
    L = _rsf_pick(rsf_params.L, phase)
    D = _rsf_pick(rsf_params.D, phase)
    λ = _rsf_pick(rsf_params.λ, phase)
    V0 = hasproperty(rsf_params, :V0) ? _rsf_pick(rsf_params.V0, phase) : rsf_params.V0_model
    Vp_raw = maximum(Array(rsf_state.Vp_center))
    # Floor slip rate so Courant / weakening limits stay finite (same spirit as first RSF substeps).
    Vp = max(Vp_raw, 1.0e-6 * V0)
    Ω = minimum(Array(rsf_state.Ω_center))
    Pm = maximum(Array(P))
    θmax = max_state_change_rsf(a, b, L, D, λ, Pm, G, ν)
    dt_c = dt_courant_rsf(Vp, D; f = f_courant)
    dt_h = dt_healing_rsf(L, V0, Ω)
    dt_w = dt_weakening_rsf(L, θmax, Vp)
    dt = min(dt_c, dt_h, dt_w)
    # return clamp(dt, dt_min, dt_max)
    return min((1/Vp)/8.0,5e8)
end

function compute_mu_eff_field(τII::AbstractMatrix, P::AbstractMatrix, phase_center, rsf_params)
    nx, ny = size(τII)
    μeff = zeros(eltype(τII), nx, ny)
    Vp = zeros(eltype(τII), nx, ny)
    rsf_params === nothing && return μeff, Vp

    @inbounds for j in 1:ny, i in 1:nx
        ratios = phase_center[i, j]
        phase = argmax(ratios)
        hasproperty(rsf_params, :active) && !_rsf_pick(rsf_params.active, phase) && continue
        if hasproperty(rsf_params, :mu_s) && hasproperty(rsf_params, :mu_d)
            μs = _rsf_pick(rsf_params.mu_s, phase)
            μd = clamp(_rsf_pick(rsf_params.mu_d, phase), 0.0, μs)
            σc = _rsf_pick(rsf_params.sigma_c, phase)
            Vc = _rsf_pick(rsf_params.Vc, phase)
            p_shift = hasproperty(rsf_params, :p_shift) ? _rsf_pick(rsf_params.p_shift, phase) : 0.0
            P_eff = P[i, j] + p_shift
            P_eff <= 0.0 && continue

            # Recover μ_eff from local stress state: τy = P_eff*μ_eff + σc.
            μ_state = (τII[i, j] - σc) / P_eff
            μeff[i, j] = clamp(μ_state, μd, μs)

            # Diagnostic Vp inferred from RSF law inverse (clamped near μd).
            denom = max(μeff[i, j] - μd, sqrt(eps(eltype(τII))))
            Vp[i, j] = max(Vc * ((μs - μd) / denom - 1.0), 0.0)
        else
            # Rate-and-state viscous diagnostics (LaMEM-like schema).
            μ0 = _rsf_pick(rsf_params.μ0, phase)
            a = _rsf_pick(rsf_params.a, phase)
            b = _rsf_pick(rsf_params.b, phase)
            λ = _rsf_pick(rsf_params.λ, phase)
            C = _rsf_pick(rsf_params.C, phase)
            V0 = hasproperty(rsf_params, :V0) ? _rsf_pick(rsf_params.V0, phase) : rsf_params.V0_model
            Ω = _rsf_pick(rsf_params.Ω_init, phase)
            p_shift = hasproperty(rsf_params, :p_shift) ? _rsf_pick(rsf_params.p_shift, phase) : 0.0
            P_eff = P[i, j] * (1.0 - λ) + p_shift
            P_eff <= 0.0 && continue

            μ_state = max((τII[i, j] - C) / P_eff, 0.0)
            μeff[i, j] = μ_state
            arg = μ_state / max(a, eps(eltype(τII)))
            exp_arg = (μ0 + b * Ω) / max(a, eps(eltype(τII)))
            Vp_loc = 2.0 * V0 * sinh(min(arg, 700.0)) * exp(-min(exp_arg, 700.0))
            Vp[i, j] = clamp(Vp_loc, 0.0, hasproperty(rsf_params, :Vp_max) ? _rsf_pick(rsf_params.Vp_max, phase) : 1.0e19)
        end
    end
    return μeff, Vp
end

function sanitize_particle_coords!(particles, xvi; periodic_x = false)
    ppx = particles.coords[1].data
    ppy = particles.coords[2].data
    idx = particles.index.data

    xmin, xmax = minimum(xvi[1]), maximum(xvi[1])
    ymin, ymax = minimum(xvi[2]), maximum(xvi[2])
    lx = xmax - xmin
    ϵx = max(eps(Float64), 1.0e-12 * lx)
    ϵy = max(eps(Float64), 1.0e-12 * (ymax - ymin))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    @parallel_indices (I) function _sanitize_particle_coords_kernel!(
            ppx, ppy, idx, xmin, xmax, ymin, ymax, lx, ϵx, ϵy, xmid, ymid, periodic_x
        )
        @inbounds if idx[I] > 0
            x = ppx[I]
            y = ppy[I]

            if !isfinite(x)
                x = xmid
            end
            if !isfinite(y)
                y = ymid
            end

            if periodic_x
                if x < xmin || x >= xmax
                    x = xmin + mod(x - xmin, lx)
                end
            else
                x = clamp(x, xmin + ϵx, xmax - ϵx)
            end

            y = clamp(y, ymin + ϵy, ymax - ϵy)

            ppx[I] = x
            ppy[I] = y
        end
        return nothing
    end

    @parallel (@idx length(idx)) _sanitize_particle_coords_kernel!(
        ppx, ppy, idx, xmin, xmax, ymin, ymax, lx, ϵx, ϵy, xmid, ymid, periodic_x
    )
    return nothing
end

@views function apply_top_shear_bc!(stokes, Vtop)
    Vx, Vy = @velocity(stokes)
    # Enforce full-width top tangential velocity and zero normal velocity.
    if isnothing(Vtop)
        return nothing
    end
    Vx[:, end] .= 2 * Vtop .- Vx[:, end - 1]
    Vy[:, end] .= 0.0
    return nothing
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
# PREPARE VISUALIZATION SETTINGS
function prepare_visualisation(ni)
    # SETTINGS FOR VISUALIZATION AND OUTPUT
    do_vtk   = true # set to true to generate VTK files for ParaView
    pictures = false # set to true to generate PNG figures of particles and fields using Makie
    # IF VTK OUTPUT YES
    pvd_name = "Simple_Shear_2d"
    figdir   = "Simple_Shear_results_rsf_unhinged"
    save_particle_points = false # set to true to save particle point clouds as VTK files (can generate large files)
    vtk_every = 1 # save VTK every N iterations
    particle_vtk_every = 1 # save particle VTK every N iterations


    if do_vtk == true
        vtk_dir = joinpath(figdir, "vtk")
        if isfile(joinpath(vtk_dir, "$pvd_name.pvd"))
            rm(joinpath(vtk_dir, "$pvd_name.pvd"))
            # preview
            # println.(filter(f->startswith(f,"vtk_"), readdir(vtk_dir)))
            # delete
            for f in filter(f->startswith(f,"vtk_"), readdir(vtk_dir))
                rm(joinpath(vtk_dir, f); force=true)
            end
        end
        take(vtk_dir)
        checkpoint = joinpath(figdir, "checkpoint")
        take(checkpoint)
    end
    quiet_runtime = false # silence non-essential console output for GPU tests
    vis=(;do_vtk,vtk_dir,pvd_name ,figdir,save_particle_points,vtk_every,particle_vtk_every,pictures,checkpoint,quiet_runtime,Vx_v = @zeros(ni .+ 1...), Vy_v = @zeros(ni .+ 1...),)

    return vis
end
# VELOCITY BOXES ROUTINES
@parallel_indices (i, j) function _apply_vel_box_Vx!(
    Vx,
    xvx,
    yvx,
    cenx,
    cenz,
    halfx,
    halfz,
    vx_val,
)
    if i ≤ size(Vx, 1) && j ≤ size(Vx, 2)
        x = xvx[i]
        z = yvx[j]
        if abs(x - cenx) ≤ halfx && abs(z - cenz) ≤ halfz
            @inbounds Vx[i, j] = vx_val
        end
    end
    return nothing
end

@parallel_indices (i, j) function _apply_vel_box_Vy!(
    Vy,
    xvy,
    yvy,
    cenx,
    cenz,
    halfx,
    halfz,
    vy_val,
)
    if i ≤ size(Vy, 1) && j ≤ size(Vy, 2)
        x = xvy[i]
        z = yvy[j]
        if abs(x - cenx) ≤ halfx && abs(z - cenz) ≤ halfz
            @inbounds Vy[i, j] = vy_val
        end
    end
    return nothing
end

@parallel_indices (i, j) function _mark_vbox_mask_Vx!(
    mask_vbox_x,
    xvx,
    yvx,
    cenx,
    cenz,
    halfx,
    halfz,
)
    if i ≤ size(mask_vbox_x, 1) && j ≤ size(mask_vbox_x, 2)
        # mask indices (i,j) correspond to velocity DoFs at (i+1,j+1)
        ii = i + 1
        jj = j + 1
        if ii ≤ length(xvx) && jj ≤ length(yvx)
            x = xvx[ii]
            z = yvx[jj]
            if abs(x - cenx) ≤ halfx && abs(z - cenz) ≤ halfz
                @inbounds mask_vbox_x[i, j] = 1
            end
        end
    end
    return nothing
end

@parallel_indices (i, j) function _mark_vbox_mask_Vy!(
    mask_vbox_y,
    xvy,
    yvy,
    cenx,
    cenz,
    halfx,
    halfz,
)
    if i ≤ size(mask_vbox_y, 1) && j ≤ size(mask_vbox_y, 2)
        # mask indices (i,j) correspond to velocity DoFs at (i+1,j+1)
        ii = i + 1
        jj = j + 1
        if ii ≤ length(xvy) && jj ≤ length(yvy)
            x = xvy[ii]
            z = yvy[jj]
            if abs(x - cenx) ≤ halfx && abs(z - cenz) ≤ halfz
                @inbounds mask_vbox_y[i, j] = 1
            end
        end
    end
    return nothing
end

# Velocity boxes are applied on the same staggered coordinates as the Stokes solver.
# In the new Geometry API these coordinates are stored in `grid.xi_vel`:
# - `grid.xi_vel[1]` are the coordinates for Vx (x-face, z)
# - `grid.xi_vel[2]` are the coordinates for Vy (x, z-face)
# so the box region is applied to the correct velocity DoFs.
function apply_vel_boxes!(
    stokes,
    grid,
    boxes::Vector{VelBox2D},
)
    isempty(boxes) && return nothing

    Vx, Vy = @velocity(stokes)
    grid_vx, grid_vy = grid.xi_vel
    xvx, yvx = grid_vx
    xvy, yvy = grid_vy

    # reset velocity-box masks: 0 ⇒ no box (free)
    stokes.mask_vbox_x.mask .= 0
    stokes.mask_vbox_y.mask .= 0

    for box in boxes
        halfx = box.widthx / 2
        halfz = box.widthz / 2

        if box.has_vx
            nx = length(xvx)
            ny = length(yvx)
            @parallel (@idx (nx, ny)) _apply_vel_box_Vx!(
                Vx, xvx, yvx, box.cenx, box.cenz, halfx, halfz, box.vx
            )
            @parallel (@idx (nx, ny)) _mark_vbox_mask_Vx!(
                stokes.mask_vbox_x.mask,
                xvx,
                yvx,
                box.cenx,
                box.cenz,
                halfx,
                halfz,
            )
        end

        if box.has_vy
            nx = length(xvy)
            ny = length(yvy)
            @parallel (@idx (nx, ny)) _apply_vel_box_Vy!(
                Vy, xvy, yvy, box.cenx, box.cenz, halfx, halfz, box.vy
            )
            @parallel (@idx (nx, ny)) _mark_vbox_mask_Vy!(
                stokes.mask_vbox_y.mask,
                xvy,
                yvy,
                box.cenx,
                box.cenz,
                halfx,
                halfz,
            )
        end
    end

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------






## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(
    li,
    origin,
    phases_GMG,
    igg;
    xvi,
    xci,
    nx = 16,
    ny = 16,
    ref_grid = 0,
    periodic_x = false,
    disable_injection_when_periodic = false,
    Vtop = nothing,
    use_rsf = true,
    rsf_monitor_every = 1,
)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
# @infiltrate
    grid = Geometry(
        PTArray(backend_JR),
        xvi...,
        )

    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    di_min = min(
        min(minimum.(grid.di.center)...),
        min(minimum.(grid.di.vertex)...),
        )
    di1 = grid.di 
    # ----------------------------------------------------
    # Set flags and parameters for visualization and output and create folders for output
    vis = prepare_visualisation(ni)
    # Physical properties using GeoParams ----------------
    rheology = init_rheology_simple_shear()
    V0_rsf = isnothing(Vtop) ? 4.0e-9 : abs(Vtop)
    rsf_params = use_rsf ? init_rsf_params_lamem_simple_shear(di_min; V0_model = V0_rsf) : nothing
    # dt = 25.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    dt = 1e8
    dt_max = 25.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 40
    max_xcell = 60
    min_xcell = 20
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    grid_vxi_raw = velocity_grids(xci, xvi, di)
    grid_vxi = ntuple(Val(length(grid_vxi_raw))) do i
        Base.@_inline_meta
        ntuple(Val(length(grid_vxi_raw[i]))) do j
            Base.@_inline_meta
            PTArray(backend)(collect(grid_vxi_raw[i][j]))
        end
    end
    # material phase & temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    periodic_x && enforce_periodic_phase_ratios_x!(phase_ratios)
    rsf_state = use_rsf ? init_rsf_state_fields(ni, phase_ratios, rsf_params) : nothing
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop = 20 + 273
    Tbot = maximum(T_GMG)
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)
    thermal_no_flux = periodic_x ?
        (left = false, right = false, top = false, bot = false) :
        (left = true, right = true, top = false, bot = false)
    thermal_bc = TemperatureBoundaryConditions(; no_flux = thermal_no_flux, periodic_x = periodic_x)
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1] .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    ρg[2] .= 0  # Turn off gravity for simple shear test
    if ref_grid == 0
        stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))
    else
    # Lithostatic pressure integrates vertical body force using local cell dy (vertex spacing).
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* reshape(di1.vertex[2], 1, :), dims = 2), dims = 2), dims = 2))
    end
    # Pressure shift to reference value
    P_ref = 5e7  # Reference pressure in Pa
    stokes.P .+= P_ref
    # Rheology
    args0 = (T = thermal.Tc, P = stokes.P, dt = Inf, periodic_x = periodic_x)
    viscosity_cutoff = (1.0e1, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)
    center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)
    # ----------------------------------------------------

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di1.center, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    flow_no_slip = periodic_x ?
        (left = false, right = false, top = false, bot = true) :
        (left = false, right = false, top = false, bot = true)
    flow_free_slip = periodic_x ?
        (left = false, right = false, top = true, bot = false) :
        (left = true, right = true, top = true, bot = false)
    flow_bcs = VelocityBoundaryConditions(;
        no_slip = flow_no_slip,
        free_slip = flow_free_slip,
        free_surface = false,
        periodic_x = periodic_x,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    stokes.mask_vbox_x.mask .= 0.0
    stokes.mask_vbox_y.mask .= 0.0
    apply_top_shear_bc!(stokes, Vtop)
    update_halo!(@velocity(stokes)...)

    ## Boundary conditions
    #     εbg = 1.0e-14 # background strain-rate
    # flow_bcs = VelocityBoundaryConditions(;
    #     free_slip = (left = true, right = true, top = true, bot = true),
    #     no_slip = (left = false, right = false, top = false, bot = false),
    # )
    # stokes.V.Vx .= PTArray(backend_JR)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    # stokes.V.Vy .= PTArray(backend_JR)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    # flow_bcs!(stokes, flow_bcs)
    # update_halo!(@velocity(stokes)...)




    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, T_buffer, particles)

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, di1, dt; ϵ = 1.0e-3, periodic_x = periodic_x)

    # Time loop
    t, it = 0.0, 0
    while it < 10000 #000 # run only for 5 Myrs
        periodic_x && wrap_particles_x!(particles, xvi)

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)

        # Prescribe top shear boundary before solve.
        # apply_vel_boxes!(stokes, grid, vel_boxes_2D)
        # apply_top_shear_bc!(stokes, Vtop)
        # update_halo!(@velocity(stokes)...)

        # Stokes solver ----------------
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        t_stokes = @elapsed begin
            out = solve_DYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    verbose_PH = !vis.quiet_runtime,
                    verbose_DR = !vis.quiet_runtime,
                    iterMax = 50.0e2,
                    rel_drop = 1.0e-2,
                    nout = 400,
                    λ_relaxation_PH = 1,
                    λ_relaxation_DR = 1,
                    viscosity_relaxation = 1.0e-2,
                    # Re-apply BCs each nonlinear iteration; no velocity boxes yet.
                    apply_velocity_box = stokes -> begin
                        flow_bcs!(stokes, flow_bcs)
                        apply_top_shear_bc!(stokes, Vtop)
                    end,
                    rsf_params = rsf_params,
                    rsf_state = rsf_state,
                    viscosity_cutoff = (1.0e1, 1.0e23),
                )
            )
        end
        # print some stuff
        if !vis.quiet_runtime
            println("Stokes solver time             ")
            println("   Total time:      $t_stokes s")
        end
        # println("   Time/iteration:  $(t_stokes / out.iter) s")

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        # compute time step
        dt = if use_rsf && rsf_params !== nothing && rsf_state !== nothing
            compute_dt_ratestate_global(rsf_params, rsf_state, stokes.P; dt_min = 1.0e-1, dt_max = dt_max)
        else
            1e8
        end
        if !vis.quiet_runtime
            println("Time step: $dt s")
            if use_rsf && rsf_state !== nothing && (it == 0 || rem(it, rsf_monitor_every) == 0)
                Ωc = Array(rsf_state.Ω_center)
                Vpc = Array(rsf_state.Vp_center)
                println("RSF monitor: Ω[min,max]=($(minimum(Ωc)), $(maximum(Ωc)))  Vp[min,max]=($(minimum(Vpc)), $(maximum(Vpc)))")
            end
        end
       
        # # compute strain rate 2nd invartian - for plotting
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        # # ------------------------------

        # # Thermal solver ---------------
        # heatdiffusion_PT!(
        #     thermal,
        #     pt_thermal,
        #     thermal_bc,
        #     rheology,
        #     args,
        #     dt,
        #     grid;
        #     kwargs = (
        #         igg = igg,
        #         phase = phase_ratios,
        #         iterMax = 50.0e3,
        #         nout = 1.0e2,
        #         verbose = true,
        #     )
        # )
        # subgrid_characteristic_time!(
        #     subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        # )
        # centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        # subgrid_diffusion!(
        #     pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt
        # )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt, particles.di.velocity)
        sanitize_particle_coords!(particles, xvi; periodic_x = periodic_x)
        periodic_x && wrap_particles_x!(particles, xvi)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        # need stresses on the vertices for injection purposes
        # center2vertex!(τxx_v, stokes.τ.xx)
        # center2vertex!(τyy_v, stokes.τ.yy)
        if !(periodic_x && disable_injection_when_periodic)
            inject_particles_phase!(
                particles,
                pPhases,
                particle_args_reduced,
                (T_buffer, stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.ω.xy)
            )
        end

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)
        periodic_x && enforce_periodic_phase_ratios_x!(phase_ratios)

        it += 1
        if !vis.quiet_runtime
            println("========================================")
            println("    Timestep $it")
            println("    Time = $(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs")
            println("=========================================")
        end
        t += dt 

        if !vis.quiet_runtime
            println("Max τII = $(mean(stokes.τ.II)) Pa")
        end
        
        ### PARAVIEW PLOTTING
        if it >= 0 #it == 1 || rem(it, 5) == 0

        

            checkpointing_jld2(vis.checkpoint, stokes, thermal, t, dt; it = it)
            checkpointing_particles(vis.checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, particle_args = particle_args, particle_args_reduced = particle_args_reduced, t = t, dt = dt, it = it)
            (; η_vep, η) = stokes.viscosity
            if vis.do_vtk && (it == 1 || rem(it, vis.vtk_every) == 0)
                Vx_v = vis.Vx_v
                Vy_v = vis.Vy_v
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                # Reconstruct compact phase "shapes" on the grid from particle phase ratios.
                phase_vertex = [argmax(p) for p in Array(phase_ratios.vertex)]

                data_v = (;
                    T = Array(T_buffer),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                    Ω = rsf_state === nothing ? zeros(size(Array(Vx_v))) : Array(rsf_state.Ω_vertex),
                    Vp = rsf_state === nothing ? zeros(size(Array(Vx_v))) : Array(rsf_state.Vp_vertex),
                    phase_vertex = phase_vertex,
                )
                RP_c = Array(stokes.R.RP)
                Rx_c, Ry_c, Rmag_c = residuals_to_center(Array(stokes.R.Rx), Array(stokes.R.Ry), RP_c)
                μ_eff_c, Vp_c = compute_mu_eff_field(Array(stokes.τ.II), Array(stokes.P), Array(phase_ratios.center), rsf_params)
                data_c = (;
                    P   = Array(stokes.P),
                    η_vep   = Array(η_vep),
                    η   = Array(η),
                    Ω = rsf_state === nothing ? zeros(size(Array(stokes.P))) : Array(rsf_state.Ω_center),
                    Vp_rsf = rsf_state === nothing ? zeros(size(Array(stokes.P))) : Array(rsf_state.Vp_center),
                    RP  = RP_c,
                    Rx_c = Rx_c,
                    Ry_c = Ry_c,
                    R_magnitude = Rmag_c,
                    μ_eff = μ_eff_c,
                    Vp = Vp_c,
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                path_vtk = joinpath(vis.vtk_dir, "vtk_" * lpad("$it", 6, "0"))
                save_vtk(
                    path_vtk,
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v;
                    t = t,
                    pvd=joinpath(vis.vtk_dir, vis.pvd_name)
                )
                if !vis.quiet_runtime
                    println("Saved VTK file at $(joinpath(vis.vtk_dir, vis.pvd_name))")
                end
                # Optional particle point-cloud output (large files).
                if vis.save_particle_points && (it == 1 || rem(it, vis.particle_vtk_every) == 0)
                    save_particles(
                        particles,
                        pPhases;
                        fname = joinpath(vis.vtk_dir, "particles_" * lpad("$it", 6, "0")),
                        t = t,
                    )
                end
                

            end

            if vis.pictures == true
            # Make particles plottable
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:] ./ 1.0e3
            pyv = ppy.data[:] ./ 1.0e3
            clr = pPhases.data[:]
            # clr      = pT.data[:]
            idxv = particles.index.data[:]
            
            # Make Makie figure
            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.T[2:(end - 1), :]), colormap = :batlow)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 1)
            # Plot 2nd invariant of strain rate
            # h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array((stokes.τ.II)), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(vis.fig_dir, "$(it).png"), fig)
            end
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------

# MODEL SETUP
n = 64
nx, ny = n * 1, n
# Choose grid type: original uniform grid (ref_grid=0) or non-uniform logistic grid (ref_grid=1)
ref_grid = 0 # 0: original uniform grid, 1: non-uniform logistic grid
periodic_x = true
disable_injection_when_periodic = false
Vtop = 4.0e-9
use_rsf = true
rsf_monitor_every = 1

# GENERATE GRID
li, origin, phases_GMG, T_GMG, xvi, xci = GMG_subduction_2D_with_coords(
    nx + 1,
    ny + 1;
    ref_grid = ref_grid,
)

# Initialize MPI grid (or not)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(
    li,
    origin,
    phases_GMG,
    igg;
    xvi,
    xci,
    nx = nx,
    ny = ny,
    ref_grid = ref_grid,
    periodic_x = periodic_x,
    disable_injection_when_periodic = disable_injection_when_periodic,
    Vtop = Vtop,
    use_rsf = use_rsf,
    rsf_monitor_every = rsf_monitor_every,
);

# julia --project=. -e 'include("miniapps/DYREL2D/iskander/new_simple_shear_rsf/SShear2D_DYREL.jl")'