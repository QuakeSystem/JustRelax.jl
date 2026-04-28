# Load script dependencies
using GeoParams, Printf#, CairoMakie

const isCUDA = false

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

# Load file with all the rheology configurations
include("Subduction2D_setup.jl")
include("Subduction2D_rheology.jl")

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
    xmin = xvi[1][1]
    xmax = xvi[1][end]
    lx = xmax - xmin
    dx = xvi[1][2] - xvi[1][1]
    ppx = particles.coords[1].data
    idx = particles.index.data

    nwrap_left = 0
    nwrap_right = 0
    nlt = 0
    nge = 0
    nnear_right = 0
    nnear_left = 0
    xmin_seen = Inf
    xmax_seen = -Inf

    @inbounds for k in eachindex(idx)
        if idx[k]
            x = ppx[k]
            xmin_seen = min(xmin_seen, x)
            xmax_seen = max(xmax_seen, x)
            (x > xmax - 0.25dx) && (nnear_right += 1)
            (x < xmin + 0.25dx) && (nnear_left += 1)
            if x < xmin
                nlt += 1
                ppx[k] = x + lx
                nwrap_left += 1
            elseif x >= xmax
                nge += 1
                ppx[k] = x - lx
                nwrap_right += 1
            end
        end
    end

    return (;
        xmin,
        xmax,
        dx,
        xmin_seen,
        xmax_seen,
        n_out_left = nlt,
        n_out_right = nge,
        n_near_left = nnear_left,
        n_near_right = nnear_right,
        nwrap_left,
        nwrap_right,
    )
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

@inline function periodicize_x_output_seam!(A::AbstractMatrix)
    # Output/diagnostic helper for x-periodic fields stored on vertex-like grids:
    # enforce seam identity on the duplicated right edge.
    @views A[end, :] .= A[1, :]
    return A
end

function sanitize_particle_coords!(particles, xvi; periodic_x = false)
    ppx = particles.coords[1].data
    ppy = particles.coords[2].data
    idx = particles.index.data

    xmin, xmax = xvi[1][1], xvi[1][end]
    ymin, ymax = xvi[2][1], xvi[2][end]
    lx = xmax - xmin
    ϵx = max(eps(Float64), 1.0e-12 * lx)
    ϵy = max(eps(Float64), 1.0e-12 * (ymax - ymin))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    nfix_nonfinite_x = 0
    nfix_nonfinite_y = 0
    nfix_wrap_x = 0
    nfix_clamp_y = 0

    for I in eachindex(idx)
        idx[I] <= 0 && continue

        x = ppx[I]
        y = ppy[I]

        if !isfinite(x)
            x = xmid
            nfix_nonfinite_x += 1
        end
        if !isfinite(y)
            y = ymid
            nfix_nonfinite_y += 1
        end

        if periodic_x
            if x < xmin || x ≥ xmax
                x = xmin + mod(x - xmin, lx)
                nfix_wrap_x += 1
            end
        else
            x = clamp(x, xmin + ϵx, xmax - ϵx)
        end

        y_clamped = clamp(y, ymin + ϵy, ymax - ϵy)
        if y_clamped != y
            y = y_clamped
            nfix_clamp_y += 1
        end

        ppx[I] = x
        ppy[I] = y
    end

    return (;
        nfix_nonfinite_x,
        nfix_nonfinite_y,
        nfix_wrap_x,
        nfix_clamp_y,
    )
end

function sanitize_particle_fields!(fields::Tuple, particles; fill_value = 0.0)
    idx = particles.index.data
    nfixed = 0
    for fld in fields
        data = fld.data
        for I in eachindex(idx)
            idx[I] <= 0 && continue
            v = data[I]
            if !isfinite(v)
                data[I] = fill_value
                nfixed += 1
            end
        end
    end
    return nfixed
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
    pvd_name = "Subduction2d"
    figdir   = "Subduction2D_uniform_grid_resultsHR"
    save_particle_points = false # set to true to save particle point clouds as VTK files (can generate large files)
    vtk_every = 1 # save VTK every N iterations
    particle_vtk_every = 1 # save particle VTK every N iterations


    if do_vtk == true
        vtk_dir = joinpath(figdir, "vtk")
        if isfile(joinpath(vtk_dir, "$pvd_name.pvd"))
            rm(joinpath(vtk_dir, "$pvd_name.pvd"))
        end
        take(vtk_dir)
        checkpoint = joinpath(figdir, "checkpoint")
        take(checkpoint)
    end
    vis=(;do_vtk,vtk_dir,pvd_name ,figdir,save_particle_points,vtk_every,particle_vtk_every,pictures,checkpoint,Vx_v = @zeros(ni .+ 1...), Vy_v = @zeros(ni .+ 1...),)

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
)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps

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
    rheology = init_rheology_nonNewtonian_plastic()
    dt = 25.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
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
    # JustPIC advection expects each staggered grid tuple to have homogeneous
    # coordinate container types; enforce Vector on all axes.
    grid_vxi = ntuple(Val(length(grid_vxi_raw))) do i
        Base.@_inline_meta
        ntuple(Val(length(grid_vxi_raw[i]))) do j
            Base.@_inline_meta
            PTArray(backend_JP)(collect(grid_vxi_raw[i][j]))
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
    if ref_grid == 0
        stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))
    else
    # Lithostatic pressure integrates vertical body force using local cell dy (vertex spacing).
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* reshape(di1.vertex[2], 1, :), dims = 2), dims = 2), dims = 2))
    end
    # Rheology
    args0 = (T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)
    center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)
    # ----------------------------------------------------

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di1, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    flow_no_slip = periodic_x ?
        (left = false, right = false, top = false, bot = true) :
        (left = false, right = false, top = false, bot = false)
    flow_free_slip = periodic_x ?
        (left = false, right = false, top = true, bot = false) :
        (left = true, right = true, top = true, bot = true)
    flow_bcs = VelocityBoundaryConditions(;
        no_slip = flow_no_slip,
        free_slip = flow_free_slip,
        free_surface = false,
        periodic_x = periodic_x,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

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
    while it < 100 #000 # run only for 5 Myrs

        # Keep particles inside periodic x-domain before any particle->grid operations.
        periodic_x && wrap_particles_x!(particles, xvi)

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:(end - 1), :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # interpolate stress back to the grid
        nfix_pτ_pre = sanitize_particle_fields!(unwrap(pτ), particles; fill_value = 0.0)
        if nfix_pτ_pre > 0 && igg.me == 0
            @printf("Sanitized non-finite particle stress/vorticity values before stress2grid: %d\n", nfix_pτ_pre)
        end
        stress2grid!(stokes, pτ, particles)
        # Prescribe velocity boxes before solve so solver finds a solution consistent with them
        apply_vel_boxes!(stokes, grid, vel_boxes_2D)
        update_halo!(@velocity(stokes)...)

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
                    verbose_PH = true,
                    verbose_DR = true,
                    iterMax = 50.0e2,
                    rel_drop = 1.0e-2,
                    nout = 400,
                    λ_relaxation_PH = 1,
                    λ_relaxation_DR = 1,
                    viscosity_relaxation = 1.0e-2,
                    apply_velocity_box = stokes -> apply_vel_boxes!(stokes, grid, vel_boxes_2D),
                    viscosity_cutoff = (1.0e18, 1.0e23),
                )
            )
        end
        # print some stuff
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        nfix_pτ_post = sanitize_particle_fields!(unwrap(pτ), particles; fill_value = 0.0)
        if nfix_pτ_post > 0 && igg.me == 0
            @printf("Sanitized non-finite particle stress/vorticity values after rotate_stress!: %d\n", nfix_pτ_post)
        end
        # compute time step
        dt = compute_dt(stokes, di_min, dt_max) #* 0.8
        # compute strain rate 2nd invartian - for plotting
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (
                igg = igg,
                phase = phase_ratios,
                iterMax = 50.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt
        )
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
        elseif it == 0
            @info "Periodic test mode: inject_particles_phase! disabled"
        end

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)
        periodic_x && enforce_periodic_phase_ratios_x!(phase_ratios)

        @show it += 1
        t += dt

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
                τxy_viz = Array(stokes.τ.xy)
                if periodic_x
                    periodicize_x_output_seam!(phase_vertex)
                    periodicize_x_output_seam!(τxy_viz)
                end

                data_v = (;
                    T = Array(T_buffer),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    τxy = τxy_viz,
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                    phase_vertex = phase_vertex,
                )
                RP_c = Array(stokes.R.RP)
                Rx_c, Ry_c, Rmag_c = residuals_to_center(Array(stokes.R.Rx), Array(stokes.R.Ry), RP_c)
                data_c = (;
                    P   = Array(stokes.P),
                    η   = Array(η_vep),
                    RP  = RP_c,
                    Rx_c = Rx_c,
                    Ry_c = Ry_c,
                    R_magnitude = Rmag_c,
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
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(ηvep)")
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
nx, ny = n * 2, n
# Choose grid type: original uniform grid (ref_grid=0) or non-uniform logistic grid (ref_grid=1)
ref_grid = 0 # 0: original uniform grid, 1: non-uniform logistic grid
periodic_x = true
disable_injection_when_periodic = false

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
);
