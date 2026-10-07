#=
MVP: asymmetric non-uniform z-grid + JustPIC's add_periodic_ghost_nodes.
Diff vs original: non-uniform z-axis (geometric_stretch_1d), x stays uniform;
ghost-node check after init_particles; thermal.T extrema printed per iteration.
=#

const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg; Pkg.activate("miniapps")
# const backend = @static if isCUDA
#     JustRelax.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# else
#     JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# end
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

using JustPIC
# const backend_JP = @static if isCUDA
#     CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
# else
#     JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
# end
const backend_JP = @static if isCUDA
    CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end


# Load script dependencies
using GeoParams, CairoMakie

# Load file with all the rheology configurations
include(joinpath(@__DIR__, "Subduction2D_setup.jl"))
include(joinpath(@__DIR__, "Subduction2D_rheology.jl"))

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

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

"""
    geometric_stretch_1d(x0, x1, dz_bottom, dz_top)

Vertices on `[x0, x1]`, cell width shrinking geometrically from `dz_bottom`
(at `x0`) to `dz_top` (at `x1`), both hit exactly, with `x0`/`x1` also hit
exactly. The cell count is searched for the geometric sum closest to `x1-x0`;
the small leftover is absorbed into one interior cell near the middle, far
from either end, so it doesn't affect the end widths or the domain extent.
"""
function geometric_stretch_1d(x0::Float64, x1::Float64, dz_bottom::Float64, dz_top::Float64)
    span = x1 - x0
    ratio = dz_bottom / dz_top
    best_n, best_leftover = 2, Inf
    for n in 2:4000
        r = ratio^(-1 / (n - 1))
        leftover = span - dz_bottom * (1 - r^n) / (1 - r)
        if abs(leftover) < abs(best_leftover)
            best_n, best_leftover = n, leftover
        end
    end
    n = best_n
    r = ratio^(-1 / (n - 1))
    widths = [dz_bottom * r^i for i in 0:(n - 1)]
    widths[(n + 1) ÷ 2] += best_leftover
    vertices = zeros(n + 1)
    vertices[1] = x0
    for i in 1:n
        vertices[i + 1] = vertices[i] + widths[i]
    end
    return vertices
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, igg, zv; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells

    # z: coarse at bottom, fine at top (zv precomputed, see bottom of file). x: uniform.
    xv = collect(range(origin[1], origin[1] + li[1], nx + 1))
    grid = Geometry(PTArray(backend), xv, zv)
    di_min = min(
        min(minimum.(grid.di.center)...),
        min(minimum.(grid.di.vertex)...),
    )
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # rheology = init_rheology_nonNewtonian_plastic()
    rheology = init_rheology_linear()
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

    # DIAGNOSTIC: ghost-node placement check (z-axis)
    let
        zvi_physical = Array(grid.xvi[2])
        dz_bottom_physical = zvi_physical[2] - zvi_physical[1]
        dz_top_physical = zvi_physical[end] - zvi_physical[end - 1]
        zvi_particles = Array(particles.xvi[2])
        dz_bottom_ghost = zvi_particles[2] - zvi_particles[1]
        dz_top_ghost = zvi_particles[end] - zvi_particles[end - 1]
        z_bottom_vertex_correct = zvi_physical[1] - dz_bottom_physical
        z_top_vertex_correct = zvi_physical[end] + dz_top_physical

        xci_physical = Array(grid.xci[2])
        dz_bottom_physical_c = xci_physical[2] - xci_physical[1]
        dz_top_physical_c = xci_physical[end] - xci_physical[end - 1]
        xci_particles = Array(particles.xci[2])
        z_bottom_centroid_correct = xci_physical[1] - dz_bottom_physical_c
        z_top_centroid_correct = xci_physical[end] + dz_top_physical_c

        println("="^70)
        println("Asymmetrically refined grid. Top is refined, bottom is coarse.")
        println("GHOST-NODE CHECK (z): dz bottom=$(dz_bottom_physical/1e3)km top=$(dz_top_physical/1e3)km")
        println("")
        println("vertex zcoord top:    ghost_used=$(zvi_particles[end]/1e3)km - ghost_expected=$(z_top_vertex_correct/1e3)km - domain_edge=$(zvi_physical[end]/1e3)km")
        println("vertex zcoord bottom: ghost_used=$(zvi_particles[1]/1e3)km - ghost_expected=$(z_bottom_vertex_correct/1e3)km - domain_edge=$(zvi_physical[1]/1e3)km")
        println("")
        println("centroid zcoord top:    ghost_used=$(xci_particles[end]/1e3)km - ghost_expected=$(z_top_centroid_correct/1e3)km - last_real_centroid=$(xci_physical[end]/1e3)km")
        println("centroid bottom: ghost_used=$(xci_particles[1]/1e3)km - ghost_expected=$(z_bottom_centroid_correct/1e3)km - first_real_centroid=$(xci_physical[1]/1e3)km")
        println("")
        println("=> bottom ghost dz: $(round((dz_bottom_physical - dz_bottom_ghost)/1e3, digits=2))km too close; particles with z<$(xci_particles[1]/1e3)km get extrapolated, not interpolated")
        println("="^70)
    end

    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    grid_vxi = velocity_grids(xci, xvi, grid.di.vertex)
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
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop = 20 + 273
    Tbot = maximum(T_GMG)
    thermal = ThermalArrays(backend, ni)
    vertex2center!(thermal.T, PTArray(backend)(T_GMG); ghost_x = true, ghost_y = true)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
    )
    thermal_bcs!(thermal, thermal_bc)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    compute_lithostatic_pressure!(stokes.P, ρg[2], di_min, igg)

    # Rheology
    args0 = (T = thermal.T, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)
    center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)
    # ----------------------------------------------------

    # PT coefficients for thermal diffusion
    # di arg needs a per-axis tuple; (di_min,di_min) stands in for non-uniform di.
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, (di_min, di_min), li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    checkpoint = joinpath(figdir, "checkpoint")
    take(checkpoint)
    # ----------------------------------------------------

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    T_buffer = thermal.T[2:(end - 1), 2:(end - 1)]
    centroid2particle!(pT, thermal.T, particles)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-3)

    dt₀ = similar(thermal.T)

    # Time loop
    t, it = 0.0, 0
    while it < 300
        Tex = extrema(Array(thermal.T))
        println("it=$it  thermal.T extrema = $Tex  finite = $(all(isfinite, Tex))")

        # interpolate fields from particles to centroids
        particle2centroid!(T_buffer, pT, particles; ghost_1 = false, ghost_2 = false, ghost_3 = false)
        @views thermal.T[2:(end - 1), 2:(end - 1)] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)

        # Stokes solver ----------------
        args = (; T = thermal.T, P = stokes.P, dt = Inf)
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
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 50.0e3,
                    rel_drop = 1.0e-2,
                    nout = 400,
                    λ_relaxation_PH = 1,
                    λ_relaxation_DR = 1,
                    viscosity_relaxation = 1.0e-2,
                    viscosity_cutoff = (1.0e18, 1.0e23),
                )
            )
        end
        # print some stuff
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        # compute time step
        dt = compute_dt(stokes, di_min, dt_max)
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
                verbose = false,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        # Populate the ghost cells before interpolating to particles.
        @views dt₀[1, :] .= dt₀[2, :]
        @views dt₀[end, :] .= dt₀[end - 1, :]
        @views dt₀[:, 1] .= dt₀[:, 2]
        @views dt₀[:, end] .= dt₀[:, end - 1]
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # Inject phase labels first, then initialize every newly injected particle field
        # through the regular centroid/vertex interpolation paths.
        inject_particles_phase!(particles, pPhases, (), ())
        centroid2particle!(pT, thermal.T, particles)
        centroid2particle!(pτ.τ_normal[1], stokes.τ.xx, particles)
        centroid2particle!(pτ.τ_normal[2], stokes.τ.yy, particles)
        grid2particle!(pτ.τ_shear[1], stokes.τ.xy, particles; ghost_1 = false, ghost_2 = false)
        grid2particle!(pτ.ω[1], stokes.ω.xy, particles; ghost_1 = false, ghost_2 = false)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        @show it += 1
        t += dt

        if !all(isfinite, extrema(Array(thermal.T)))
            println("!! thermal.T is no longer finite -- stopping early at it=$it !")
            break
        end

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 25) == 0
            # checkpointing_jld2(checkpoint, stokes, thermal, t, dt; it = it)
            # checkpointing_particles(checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, particle_args = particle_args, particle_args_reduced = particle_args_reduced, t = t, dt = dt, it = it)
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                )
                data_c = (;
                    P = Array(stokes.P),
                    T = Array(T_buffer),
                    η = Array(η_vep),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v;
                    t = t
                )
            end

            # Make particles plottable
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:] ./ 1.0e3
            pyv = ppy.data[:] ./ 1.0e3
            clr = pPhases.data[:]
            idxv = particles.index.data[:]

            # Make Makie figure
            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(thermal.T[2:(end - 1), 2:(end - 1)]), colormap = :batlow)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 1)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.ε.II)), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------
    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = false # set to physical to generate VTK files for ParaView
figdir = "Subduction2D_DYREL_nonuniform_mvp"
n = 32
nx = n * 2

# z0, z1 must match Subduction2D_setup.jl's model_depth/air_thickness.
# dz_bottom/dz_top are round, easy-to-read numbers; ny is derived from them.
z0, z1 = -260.0e3, 0.0
dz_bottom, dz_top = 30.0e3, 0.5e3
zv = geometric_stretch_1d(z0, z1, dz_bottom, dz_top)
ny = length(zv) - 1
println("z-grid: ny=$ny cells, dz_bottom=$(dz_bottom/1e3)km, dz_top=$(dz_top/1e3)km, realized span=$((zv[end]-zv[1])/1e3)km")

li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx + 1, ny + 1)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(li, origin, phases_GMG, igg, zv; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
