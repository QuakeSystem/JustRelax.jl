## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
using Printf

function main(
        staggered_grid, phases_GMG, T_GMG, igg;
        nx = 16, ny = 16, periodic = false, shear_rate_top = 0.0, dt = 500.0,
        rsf = nothing,
    )

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
    rsf_enabled = !isnothing(rsf) && Bool(get(rsf, :enabled, true))
    # Herrendörfer η when RSF on; otherwise keep previous high-η Maxwell buffer
    rheology = init_rheology_simple_shear(; η = rsf_enabled ? 5.0e26 : 1.0e23)
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

    # RSF controller + fields (optional)
    rsf_bundle = nothing
    if rsf_enabled
        rsf_ctrl = build_rate_state_controller(
            rsf; nphases = length(rheology), di = di_min
        )
        rsf_fields = RateStateArrays(ni; loc = rsf_ctrl.loc)
        init_rate_state_fields!(rsf_fields, rsf_ctrl, phase_ratios, xci)
        rsf_bundle = (; ctrl = rsf_ctrl, fields = rsf_fields)
    end

    # TEMPERATURE PROFILE --------------------------------
    Ttop = 20 + 273
    Tbot = maximum(T_GMG)
    thermal = ThermalArrays(backend, ni)
    vertex2center!(thermal.T, PTArray(backend)(T_GMG); ghost_x = true, ghost_y = true)
    thermal_bc = if periodic
        TemperatureBoundaryConditions(;
            no_flux = (left = false, right = false, top = false, bot = false),
            constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
            periodic = (left = true, right = true, top = false, bot = false),
        )
    else
        TemperatureBoundaryConditions(;
            no_flux = (left = true, right = true, top = false, bot = false),
            constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
        )
    end
    thermal_bcs!(thermal, thermal_bc)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    # Pressure: with g=0 hydrostatic init is P≈0, but RSF needs τ = P(1-λ)μ_d.
    # LaMEM Shear_test_PBC_herrendorfer uses p_shift = 5e6 Pa — match that when RSF is on.
    if rsf_enabled
        stokes.P .= 5.0e6
    else
        # hydrostatic init: use min Δy (conservative)
        stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di_min[2], dims = 2), dims = 2), dims = 2))
    end

    # Rheology
    # Maxwell: η_ve ≈ G*dt when η ≫ G*dt (quasi-elastic). With G=3e10, dt=500 → η_ve≈1.5e13.
    # Do NOT use η_min=1e18: that clamps η_ve UP and kills elastic loading / Couette propagation.
    args0 = (T = thermal.T, P = stokes.P, dt = dt)
    # LaMEM eta_min/eta_max for RSF shear test; keep a floor so series RSF cannot collapse η→0
    viscosity_cutoff = rsf_enabled ? (1.0e10, 5.0e26) : (1.0e10, 1.0e28)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)
    # η_vep is filled in solve!; seed it so plotting / early stress update aren't all-NaN
    stokes.viscosity.η_vep .= stokes.viscosity.η

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di_min, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Couette-like BCs: left/right periodic, bot no-slip, top free-slip + prescribed Vx
    # (matches JR_dev SShear2D_DYREL; top_Vx reapplied every PT iter inside flow_bcs!)
    flow_bcs = if periodic
        VelocityBoundaryConditions(;
            no_slip = (left = false, right = false, top = false, bot = true),
            free_slip = (left = false, right = false, top = true, bot = false),
            free_surface = false,
            periodic = (left = true, right = true, top = false, bot = false),
            prescribed = (; top_Vx = shear_rate_top),
        )
    else
        VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true),
            free_surface = false,
        )
    end
    stokes.V.Vx .= 0.0
    stokes.V.Vy .= 0.0
    # Linear Couette initial guess (helps PT propagate from top BC)
    if periodic && shear_rate_top != 0
        y_vx = Array(grid.xi_vel[1][2])
        ybot, ytop = first(y_vx), last(y_vx)
        Ly_v = ytop - ybot
        stokes.V.Vx .= PTArray(backend)([
                shear_rate_top * (y - ybot) / Ly_v for _ in Array(grid.xi_vel[1][1]), y in y_vx
            ])
    end
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(VTK.folder)
    if VTK.do_vtk
        take(VTK.vtk_dir)
        take(VTK.checkpoint_dir)
        # WriteVTK always opens PVD with append=true; wipe previous run so it starts fresh
        pvd_path = joinpath(VTK.vtk_dir, VTK.pvd_name * ".pvd")
        isfile(pvd_path) && rm(pvd_path)
        for f in readdir(VTK.vtk_dir)
            if startswith(f, "vtk_") || startswith(f, "particles_")
                rm(joinpath(VTK.vtk_dir, f); force = true)
            end
        end
    end
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

        if !isnothing(rsf_bundle)
            refresh_rsf_ab_mask!(rsf_bundle.fields, rsf_bundle.ctrl, phase_ratios, xci)
        end

        # Stokes solver ----------------
        args = (; T = thermal.T, P = stokes.P, dt = dt)
        dt_used = dt  # physical step size used in this Stokes solve (do not overwrite before t+=)
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
                dt_used,
                igg;
                kwargs = (
                    iterMax = 20.0e3,
                    nout = 1.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                    free_surface = false,
                    viscosity_relaxation = 1.0e-2,
                    rsf = rsf_bundle,
                )
            )
        end

        # ------------------------------------------------------------------
        # Timestep for the *next* physical step
        #   • this Stokes call used `dt_used` (initial prescribed default 500 s)
        #   • RSF on: if dt_rsf < dt_rsf_switch → RSF dt (clamped); else CFL
        #   • RSF off: keep the prescribed / previous dt (no CFL override)
        # ------------------------------------------------------------------
        dt_src = "prescribed"
        if !isnothing(rsf_bundle) && hasproperty(out, :dt_rsf)
            dt_cfl = compute_dt(stokes, di_min, rsf_bundle.ctrl.dt_max, igg)
            dt_rsf = out.dt_rsf
            dt_rsf_switch = if hasproperty(rsf_bundle.ctrl, :dt_rsf_switch)
                rsf_bundle.ctrl.dt_rsf_switch
            elseif !isnothing(rsf) && hasproperty(rsf, :dt_rsf_switch)
                Float64(rsf.dt_rsf_switch)
            else
                1.0e9
            end
            # LaMEM: apply RSF dt constraint only after the first physical step (istep > 1)
            if it ≥ 1 && dt_rsf < dt_rsf_switch
                dt = clamp(dt_rsf, rsf_bundle.ctrl.dt_min, rsf_bundle.ctrl.dt_max)
                dt_src = "RSF"
            elseif it ≥ 1
                dt = dt_cfl
                dt_src = "CFL"
            else
                # keep prescribed initial dt for the second Stokes step too if desired;
                # LaMEM keeps building CFL/dt_next on step 0–1 before RSF can shrink it
                dt_src = "prescribed (RSF deferred, LaMEM istep>1)"
            end
            f4(x) = @sprintf("%.4e", x)
            println("Stokes solver time             ")
            println("   Total time:      $(f4(t_stokes)) s")
            println("   Time/iteration:  $(f4(t_stokes / out.iter)) s")
            println("========================================")
            println("    Timestep $it")
            println("    Time     = $(f4(t)) sec")
            println("    dt_used  = $(f4(dt_used)) sec  (this Stokes step)")
            println("    next dt  = $(f4(dt)) sec  [$dt_src]")
            println("    dt_rsf   = $(f4(dt_rsf)) sec  (switch < $(f4(dt_rsf_switch)))")
            println("    dt_cfl   = $(f4(dt_cfl)) sec")
            Vp_max = hasproperty(out, :Vp_max) ? out.Vp_max : NaN
            dt_h = hasproperty(out, :dt_h) ? out.dt_h : NaN
            dt_w = hasproperty(out, :dt_w) ? out.dt_w : NaN
            dt_c = hasproperty(out, :dt_c) ? out.dt_c : NaN
            println("    Vp_max   = $(f4(Vp_max)) m/s")
            println("    dt_h/w/c = $(f4(dt_h)) / $(f4(dt_w)) / $(f4(dt_c)) sec")
            println("=========================================")
        else
            f4(x) = @sprintf("%.4e", x)
            println("Stokes solver time             ")
            println("   Total time:      $(f4(t_stokes)) s")
            println("   Time/iteration:  $(f4(t_stokes / out.iter)) s")
            println("========================================")
            println("    Timestep $it")
            println("    Time = $(f4(t)) sec")
            println("    dt   = $(f4(dt_used)) sec  [$dt_src]")
            println("=========================================")
        end
        # rotate stresses with the dt that was used for this physical step
        rotate_stress!(pτ, stokes, particles, dt_used)
        # compute strain rate 2nd invartian - for plotting
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        # ------------------------------

        # Thermal solver ---------------
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(
            pT, T_buffer, thermal.ΔT, subgrid_arrays, particles, dt_used
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt_used)
        periodic && wrap_particles_x!(particles, xvi)
        # advect particles in memory
        move_particles!(particles, particle_args)
        periodic && wrap_particles_x!(particles, xvi)
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
        t += dt_used

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
            if !isnothing(rsf_bundle)
                Vp_c = Array(rsf_bundle.fields.Vp)
                data_c = merge(
                    data_c,
                    (;
                        Ω = Array(rsf_bundle.fields.Ω),
                        Vp = Vp_c,
                        log10_Vp = log10.(max.(Vp_c, 1.0e-30)),
                        a_eff = Array(rsf_bundle.fields.a_eff),
                        b_eff = Array(rsf_bundle.fields.b_eff),
                    ),
                )
            end
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
            checkpointing_jld2(VTK.checkpoint_dir, stokes, thermal, t, dt_used; it = it)
            checkpointing_particles(
                VTK.checkpoint_dir, particles;
                phases = pPhases,
                phase_ratios = phase_ratios,
                particle_args = particle_args,
                particle_args_reduced = particle_args_reduced,
                t = t, dt = dt_used, it = it,
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
