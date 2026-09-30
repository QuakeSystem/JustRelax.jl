
## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(staggered_grid, phases_GMG, T_GMG, igg; nx = 16, ny = 16)

    # Physical domain ------------------------------------
    (; li, xvi, xci) = staggered_grid
    ni = nx, ny
    grid = Geometry(
        PTArray(backend),
        xvi...,
    )
    # Scalar mins of local spacings for PT / CFL only (solvers use full grid.di vectors)
    di_min = minimum.(grid.di.vertex)
    # ----------------------------------------------------
    # Physical properties using GeoParams ----------------
    rheology = init_rheology_simple_shear()
    dt = 10.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 40
    max_xcell = 60
    min_xcell = 20
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    grid_vxi = velocity_grids(xci, xvi, grid.di.center)  # vector-aware overload
    # material phase & temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di_min; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, Re = 20.0e0, r = 0.7, CFL = 0.9 / √2.1)
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
    # hydrostatic init: use min Δy (conservative); for strongly varying dy prefer cumsum with grid.di.center[2]
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di_min[2], dims = 2), dims = 2), dims = 2))

    # Rheology
    args0 = (T = thermal.T, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)
    # η_vep is filled in solve!; seed it so plotting / early stress update aren't all-NaN
    stokes.viscosity.η_vep .= stokes.viscosity.η

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di_min, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Simple-shear drive: Vx = ε̇ y, Vy = 0 (kinematic field; free-slip faces)
    # Without this, V≈0 → compute_dt → Inf → advection does x+0*Inf=NaN → inject [0,0]
    εbg = 1.0e-14 # s⁻¹
    grid_vx = grid.xi_vel[1]
    stokes.V.Vx .= PTArray(backend)([εbg * y for _ in Array(grid_vx[1]), y in Array(grid_vx[2])])
    stokes.V.Vy .= 0.0
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(VTK.folder)
    VTK.do_vtk && take(VTK.vtk_dir)
    VTK.do_vtk && take(VTK.checkpoint_dir)
    VTK.pictures && take(VTK.fig_dir)
    # vertex velocity buffers for VTK export
    local Vx_v, Vy_v
    if VTK.do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end
    # ----------------------------------------------------

    T_buffer = thermal.T[2:(end - 1), 2:(end - 1)]
    dt₀ = similar(stokes.P)
    centroid2particle!(pT, T_buffer, particles)

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0

    while it < 1000 # run only for 5 Myrs

        # interpolate fields from particles to centroids
        particle2centroid!(T_buffer, pT, particles)
        @views thermal.T[2:(end - 1), 2:(end - 1)] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)

        # Stokes solver ----------------
        args = (; T = thermal.T, P = stokes.P, dt = Inf)
        t_stokes = @elapsed begin
            out = solve!(
                stokes,
                pt_stokes,
                grid,
                flow_bcs,
                ρg,
                phase_ratios,
                rheology,
                args,
                dt,
                igg;
                kwargs = (
                    iterMax = 1.0e3,
                    nout = 2.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                    free_surface = false,
                    viscosity_relaxation = 1.0e-2,
                )
            )
        end

        # print some stuff
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")
        println("========================================")
        println("    Timestep $it")
        println("    Time = $(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs")
        println("=========================================")
        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        # CFL dt; never pass Inf/NaN to advection (0*Inf → NaN coords → inject [0,0])
        dt_CFL = compute_dt(stokes, di_min) * 0.8
        if isfinite(dt_CFL) && dt_CFL > 0
            dt = dt_CFL
        end
        # compute strain rate 2nd invartian - for plotting
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        # ------------------------------

        # Thermal solver ---------------
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
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(
            pT, T_buffer, thermal.ΔT, subgrid_arrays, particles, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        # need stresses on the vertices for injection purposes
        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy)
        )

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if VTK.do_vtk && (it == 1 || rem(it, VTK.vtk_every) == 0)
            (; η_vep, η) = stokes.viscosity
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            phase_vertex = [argmax(p) for p in Array(phase_ratios.vertex)]

            data_v = (;
                Vx = Array(Vx_v),
                Vy = Array(Vy_v),
                phase = phase_vertex,
            )
            data_c = (;
                P = Array(stokes.P),
                T = Array(T_buffer),
                η = Array(η),
                η_vep = Array(η_vep),
                τII = Array(stokes.τ.II),
                εII = Array(stokes.ε.II),
                RP = Array(stokes.R.RP),
            )
            velocity_v = (Array(Vx_v), Array(Vy_v))
            path_vtk = joinpath(VTK.vtk_dir, "vtk_" * lpad("$it", 6, "0"))
            save_vtk(
                path_vtk,
                xvi,
                xci,
                data_v,
                data_c,
                velocity_v;
                t = t,
                pvd = joinpath(VTK.vtk_dir, VTK.pvd_name),
            )
            if !VTK.quiet_runtime
                println("Saved VTK → $(joinpath(VTK.vtk_dir, VTK.pvd_name)).pvd  (it=$it)")
            end
            if VTK.save_particle_points && (it == 1 || rem(it, VTK.particle_vtk_every) == 0)
                save_particles(
                    particles,
                    pPhases;
                    fname = joinpath(VTK.vtk_dir, "particles_" * lpad("$it", 6, "0")),
                    t = t,
                )
            end
            checkpointing_jld2(VTK.checkpoint_dir, stokes, thermal, t, dt; it = it)
            checkpointing_particles(
                VTK.checkpoint_dir, particles;
                phases = pPhases,
                phase_ratios = phase_ratios,
                particle_args = particle_args,
                particle_args_reduced = particle_args_reduced,
                t = t, dt = dt, it = it,
            )
        end

        if VTK.pictures && (it == 1 || rem(it, VTK.picture_every) == 0)
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:] ./ 1.0e3
            pyv = ppy.data[:] ./ 1.0e3
            clr = pPhases.data[:]
            idxv = particles.index.data[:]

            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "τII")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η_vep)")
            h1 = heatmap!(ax1, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(T_buffer), colormap = :batlow)
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 1)
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.τ.II), colormap = :batlow)
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(VTK.fig_dir, "$(it).png"), fig)
        end

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------