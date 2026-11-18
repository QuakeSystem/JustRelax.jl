#=
SEAS BP1-FD (Erickson et al.) on the elastic eigenstrain rate-and-state split.

Reference input: LaMEM `SEAS_BP1_FD100m_prestress.dat` (side1000, depth 74 km,
ksponly, `init_seas_tau0`). The fault is one vertex row at `y = 0`. `x` is depth,
`y` is fault-normal.

This run is a 160 km × 160 km box: depth from −160 km to the free surface,
fault-normal from −80 km to 80 km. `Wf = 100 m` is the LaMEM fault width in
`V / (2 Wf)` and the fine cell size. `Wf_height = 40 km` is the SEAS down-dip
height of the rate-and-state section. Nodes deeper than `Wf_height` keep the
plate rate and are not inverted.

The fault grid is 100 m × 100 m. Fault-normal, that band is 20 cells (2 km,
fault on a vertex). Along depth, 100 m cells run from the surface to 20 km.
Outside that band, and below 20 km, cells grow at an adjacent ratio of 3. The
largest jump, including the fine-to-coarse junction, stays under 3.6.

Periodic `y` is not used. DYREL periodic directions need uniform spacing, and a
periodic pair copies the tangential velocity from one side onto the other, so
the two plates cannot move at opposite rates. The bottom wall is fixed and the
top wall moves at the plate rate `1e-9 m/s` (`Vghost = 2 Vp − Vx_interior`,
reapplied every iteration). That is the same velocity jump as the LaMEM boxes
at `±Vp/2`.

The Earth's free surface is the right edge, `x = 0`. That `Vx` column is a
boundary face the momentum kernel does not update. It is copied from the
adjacent interior column every iteration (`∂Vx/∂x = 0`). Leaving the initial
plate-rate step on that face shears the shallow fault across one `Wf` cell and
builds stress at `G Vp / Wf`, so the surface loads on its own instead of
nucleating near 12.5 km and rupturing upward.

The fault starts at the SEAS prestress: uniform `τ0` from `a_max`, `V = Vp`,
and `ψ(x)` such that the local `a(x)` returns that same stress and rate.

After each elastic solve the slip rate is the root of the sinh law on
`τ − (G Δt / Wf)(V − V_lag)`, not the law at the frozen traction. Up to 8
passes are allowed, because the onset of slip needed 6 before the rate settled.
Each of those solves starts with a cleared pseudo-velocity. The kernels are the
shared ones included by the CUDA and AMDGPU modules; leave `isCUDA = false`
here and switch it on only for a device run.
=#
const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Printf

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

using GeoParams

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

const YEAR = 365.25 * 86400.0

"""
Outer cell sizes that fill `L`. Cells grow at `ratio`, and the series is scaled
so the sum is `L`. The count is the one whose sum is closest to `L` while the
fine-to-coarse jump stays below `jump_max`.
"""
function cell_sizes_ratio(L, h_fine; ratio = 3.0, jump_max = 3.5)
    ratio > 1 || error("growth ratio must be > 1, got $ratio")
    L > 0 || error("outer length must be positive, got $L")
    h1 = ratio * h_fine
    best_n, best = 0, Inf
    n = 1
    while n ≤ 60
        s = h1 * (ratio^n - 1) / (ratio - 1)
        h_first = h1 * L / s
        r = max(h_first / h_fine, h_fine / h_first)
        if r < jump_max
            err = abs(log(s / L))
            if err < best
                best = err
                best_n = n
            end
        end
        s > L * jump_max && best_n > 0 && break
        n += 1
    end
    best_n > 0 || error(
        "cannot fill $L m from $h_fine m at ratio $ratio with jump < $jump_max",
    )
    scale = L / (h1 * (ratio^best_n - 1) / (ratio - 1))
    return [h1 * ratio^(k - 1) * scale for k in 1:best_n]
end

"""
BP1-FD mesh. `x` is depth (`h_x` from `x_fine` to the surface, then ratio growth
downward). `y` is fault-normal: `n_fault` cells of `h_fault` centered on the
fault, then ratio growth to the sides.
"""
function seas_bp1_graded_mesh(;
        x0 = -160.0e3, x1 = 0.0, x_fine = -20.0e3, h_x = 100.0,
        y0 = -80.0e3, y1 = 80.0e3, y_fault = 0.0, h_fault = 100.0, n_fault = 20,
        ratio = 3.0,
    )
    (x0 < x_fine < x1) || error("fine depth interval ($x_fine, $x1) is outside ($x0, $x1)")
    (x1 - x_fine) % h_x == 0 ||
        error("fine depth $(x1 - x_fine) m is not a multiple of $h_x m")
    n_fine_x = round(Int, (x1 - x_fine) / h_x)
    h_deep = cell_sizes_ratio(x_fine - x0, h_x; ratio = ratio)
    xv = Float64[x0]
    x = x0
    for h in reverse(h_deep)
        x += h
        push!(xv, x)
    end
    xv[end] = x_fine
    for _ in 1:n_fine_x
        x = xv[end] + h_x
        push!(xv, x)
    end
    xv[end] = x1

    n_fault > 0 && iseven(n_fault) ||
        error("fault band needs a positive even cell count so the fault sits on a vertex")
    h_fault > 0 || error("fault cell size must be positive, got $h_fault")
    fine_width = n_fault * h_fault
    y_fine0 = y_fault - fine_width / 2
    y_fine1 = y_fault + fine_width / 2
    (y0 < y_fine0 < y_fault < y_fine1 < y1) || error("fault band is outside the domain")
    h_bot = cell_sizes_ratio(y_fine0 - y0, h_fault; ratio = ratio)
    h_top = cell_sizes_ratio(y1 - y_fine1, h_fault; ratio = ratio)
    yv = Float64[y0]
    y = y0
    for h in reverse(h_bot)
        y += h
        push!(yv, y)
    end
    yv[end] = y_fine0
    for _ in 1:n_fault
        y = yv[end] + h_fault
        push!(yv, y)
    end
    yv[end] = y_fine1
    y = yv[end]
    for h in h_top
        y += h
        push!(yv, y)
    end
    yv[end] = y1

    dxs, dys = diff(xv), diff(yv)
    all(dxs .> 0) && all(dys .> 0) || error("vertices are not strictly increasing")
    ratio_x = dxs[2:end] ./ dxs[1:(end - 1)]
    ratio_y = dys[2:end] ./ dys[1:(end - 1)]
    jump = maximum(max.(vcat(ratio_x, ratio_y), 1 ./ vcat(ratio_x, ratio_y)))
    jump < 3.6 || error("adjacent cell-size ratio $jump is above 3.6")
    return (;
        xvi = (xv, yv), ni = (length(xv) - 1, length(yv) - 1),
        dx = dxs, dy = dys, ratio_x = ratio_x, ratio_y = ratio_y, jump = jump,
    )
end

function prepare_eigen_vtk!(dir, pvd_name)
    mkpath(dir)
    for f in readdir(dir)
        if startswith(f, "step_") || endswith(f, ".pvd") || endswith(f, ".vtu") ||
                endswith(f, ".vtr") || endswith(f, ".csv")
            rm(joinpath(dir, f); force = true)
        end
    end
    return joinpath(dir, pvd_name)
end

function write_eigen_vtk!(vtk_dir, pvd, it, t_s, xvi, xci, stokes, ez, Vx_v, Vy_v)
    velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
    V = Array(ez.V)
    mask = Array(ez.mask)
    logV = fill(-30.0, size(V))
    @inbounds for i in eachindex(V, mask)
        mask[i] > 0.5 && (logV[i] = log10(max(V[i], 1.0e-30)))
    end
    data_v = (;
        Vx = Array(Vx_v),
        Vy = Array(Vy_v),
        τxy = Array(stokes.τ.xy),
        V_slip = V,
        log10_V_slip = logV,
        ψ = Array(ez.ψ),
        a = Array(ez.a),
        b = Array(ez.b),
        fault = mask,
    )
    data_c = (; P = Array(stokes.P))
    path = joinpath(vtk_dir, "step_" * lpad(string(it), 6, '0'))
    save_vtk(path, xvi, xci, data_v, data_c, (Array(Vx_v), Array(Vy_v)); t = t_s, pvd = pvd)
    return nothing
end

function init_host_rheology(; η, G, ν, ρ)
    el = ConstantElasticity(; G = G, ν = ν)
    return (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η), el)),
            Elasticity = el,
        ),
    )
end

function apply_plate_velocity!(stokes, Vp)
    Vx = stokes.V.Vx
    @views Vx[:, end] .= 2 * Vp .- Vx[:, end - 1]
    return nothing
end

const BASELINE_DEPTHS = (0.0, 10.0e3, 12.5e3, 20.0e3)

"""`fault_<km>.csv` from the LaMEM baseline: time, `dev_stress_xy`, `log10(θ)`, `Vp`."""
function load_baseline_station(path)
    rows = readlines(path)
    n = length(rows) - 1
    t = Vector{Float64}(undef, n)
    τ = Vector{Float64}(undef, n)
    state = Vector{Float64}(undef, n)
    V = Vector{Float64}(undef, n)
    for (k, line) in enumerate(@view rows[2:end])
        c = split(line, ',')
        t[k] = parse(Float64, c[1])
        τ[k] = parse(Float64, c[8])
        state[k] = parse(Float64, c[22])
        V[k] = parse(Float64, c[24])
    end
    return (; t, τ, state, V)
end

function load_baseline(dir)
    series = Dict{Float64, NamedTuple}()
    for depth in BASELINE_DEPTHS
        km = depth / 1.0e3
        label = km == floor(km) ? string(Int(km)) : string(km)
        series[depth] = load_baseline_station(joinpath(dir, "fault_$(label)km.csv"))
    end
    return series
end

"""Linear stress and `log10(θ)`; slip rate is interpolated in log space."""
function baseline_at(series, t)
    ts = series.t
    if t <= ts[1]
        return (; τ = series.τ[1], state = series.state[1], V = series.V[1])
    elseif t >= ts[end]
        return (; τ = series.τ[end], state = series.state[end], V = series.V[end])
    end
    i = searchsortedlast(ts, t)
    α = (t - ts[i]) / (ts[i + 1] - ts[i])
    τ = series.τ[i] + α * (series.τ[i + 1] - series.τ[i])
    state = series.state[i] + α * (series.state[i + 1] - series.state[i])
    v0, v1 = series.V[i], series.V[i + 1]
    V = if v0 > 0 && v1 > 0
        exp(log(v0) + α * (log(v1) - log(v0)))
    else
        v0 + α * (v1 - v0)
    end
    return (; τ, state, V)
end

log10_theta(ψ, L, V₀) = ψ / log(10) + log10(L / V₀)

function main(
        igg;
        xvi,
        Wf = 100.0,
        Wf_height = 40.0e3,
        y_fault = 0.0,
        Vp = 1.0e-9,
        η0 = 5.0e26,
        G0 = 3.2038e10,
        ν = 0.25,
        ρ = 2670.0,
        σ = 50.0e6,
        a_x = (-18.0e3, -15.0e3),
        a_val = (0.025, 0.01),
        b = 0.015,
        μ₀ = 0.6,
        V₀ = 1.0e-6,
        L = 0.008,
        nsteps = 2,
        max_time = 5.8e9,
        vtk_dir = nothing,
        pvd_name = "seas_bp1_fd_eigenstrain",
        reference_dir = nothing,
    )
    ni = (length(xvi[1]) - 1, length(xvi[2]) - 1)
    grid = Geometry(PTArray(backend), xvi...)
    rheology = init_host_rheology(; η = η0, G = G0, ν = ν, ρ = ρ)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    a_max = maximum(a_val)
    τ0 = eigenstrain_steady_stress(a_max, σ, μ₀, b, V₀, Vp)
    proto = RateStateFriction(0.0, μ₀, V₀, a_val[end], b, L, 0.0, Wf)
    ez = eigenstrain_rsf(
        backend, ni; Wf = Wf, G = G0, σ = σ, law = proto, tol = 1.0e-2, max_pass = 8,
    )
    # Depth is negative downward. Creep is imposed where x <= −Wf_height.
    j_fault = init_eigenstrain_row!(
        ez, grid.xvi;
        y_fault = y_fault, a = a_val[end], b = b, ψ = 0.0, V_lag = Vp,
        a_x = a_x, a_val = a_val, x_creep = -Wf_height,
    )
    set_eigenstrain_ψ_from_stress!(ez, τ0, Vp)

    xv = Array(grid.xvi[1])
    yv = Array(grid.xvi[2])
    mask = Array(ez.mask)[:, j_fault]
    aa = Array(ez.a)[:, j_fault]
    i_surf = argmin(abs.(xv))
    i_10 = argmin(abs.(xv .+ 10.0e3))
    i_deep = argmin(abs.(xv .+ Wf_height))
    stations = NamedTuple[]
    for depth in BASELINE_DEPTHS
        i = argmin(abs.(xv .+ depth))
        push!(stations, (; depth = depth, i = i, x = xv[i]))
    end
    reference = isnothing(reference_dir) ? nothing : load_baseline(reference_dir)
    dy_lo = yv[j_fault] - yv[j_fault - 1]
    dy_hi = yv[j_fault + 1] - yv[j_fault]
    println(
        "BP1-FD fault row j=$j_fault  y=$(yv[j_fault]) m  dy=($dy_lo, $dy_hi) m  " *
            "Wf=$Wf m  Wf_height=$Wf_height m  ni=$ni",
    )
    println(
        "τ0 = $(τ0 / 1.0e6) MPa   a(surface)=$(aa[i_surf])  a(−10 km)=$(aa[i_10])  " *
            "a(−12.5 km)=$(aa[stations[3].i])  a(−20 km)=$(aa[argmin(abs.(xv .+ 20.0e3))])  a(−40 km)=$(aa[i_deep])",
    )
    (dy_lo ≈ Wf && dy_hi ≈ Wf) ||
        error("fault vertex is not between two $Wf m cells (dy = $dy_lo, $dy_hi)")
    yv[j_fault] ≈ y_fault || error("fault vertex y=$(yv[j_fault]) is not $y_fault")
    aa[i_surf] ≈ 0.01 || error("surface a=$(aa[i_surf]) is not 0.010")
    aa[argmin(abs.(xv .+ 20.0e3))] ≈ 0.025 || error("a at −20 km is not 0.025")
    any(mask[xv .≤ -Wf_height] .== 2) || error("no prescribed plate-rate nodes below Wf_height")
    any(mask[xv .> -Wf_height] .== 1) || error("no rate-and-state nodes above Wf_height")
    flush(stdout)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = false, right = false, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, τ0)
    fill!(stokes.τ_o.xy, τ0)
    yVx = Array(grid.xi_vel[1][2])
    stokes.V.Vx .= PTArray(backend)(
        [
            y ≥ y_fault ? Vp : 0.0 for _ in xvi[1], y in yVx
        ]
    )
    fill!(stokes.V.Vy, 0.0)
    apply_plate_velocity!(stokes, Vp)
    flow_bcs!(stokes, flow_bcs)
    apply_plate_velocity!(stokes, Vp)
    stokes.V0.Vx .= stokes.V.Vx
    stokes.V0.Vy .= stokes.V.Vy
    update_halo!(@velocity(stokes)...)

    dt = clamp(eigenstrain_lamem_dt(ez; ν = ν), 1.0e-6, 1.0e9)
    println("initial dt = $dt s")
    flush(stdout)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6, CFL = 0.9)

    xvi_h = Array.(grid.xvi)
    xci_h = Array.(grid.xci)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    pvd = nothing
    hist = nothing
    probe = nothing
    if !isnothing(vtk_dir)
        pvd = prepare_eigen_vtk!(vtk_dir, pvd_name)
        hist = open(joinpath(vtk_dir, "history.csv"), "w")
        println(hist, "step,t_year,t_s,passes,rel,Vmax,dt,tau_12.5km,tau_12.5km_ref,V_12.5km,V_12.5km_ref,wall_s")
        probe = open(joinpath(vtk_dir, "fault_stations.csv"), "w")
        println(probe, "Time,Dt,depth_km,x_m,Tau,Tau_ref,V,V_ref,Psi,log10_theta,log10_theta_ref")
        flush(hist)
        flush(probe)
    end

    function write_stations!(io, t_now, dt_now, τ_row, V_row, ψ_row)
        isnothing(io) && return nothing
        for s in stations
            ref = isnothing(reference) ? nothing : baseline_at(reference[s.depth], t_now)
            τ_ref = isnothing(ref) ? NaN : ref.τ
            V_ref = isnothing(ref) ? NaN : ref.V
            θ_ref = isnothing(ref) ? NaN : ref.state
            println(
                io,
                "$t_now,$dt_now,$(s.depth / 1.0e3),$(s.x),$(τ_row[s.i]),$τ_ref,$(V_row[s.i]),$V_ref,$(ψ_row[s.i]),$(log10_theta(ψ_row[s.i], L, V₀)),$θ_ref",
            )
        end
        flush(io)
        return nothing
    end

    ψ0 = Array(ez.ψ)[:, j_fault]
    V0_row = Array(ez.V)[:, j_fault]
    τ0_row = fill(τ0, length(ψ0))
    write_stations!(probe, 0.0, 0.0, τ0_row, V0_row, ψ0)

    t = 0.0
    it = 0
    last_vtk_t = -Inf
    try
        if !isnothing(pvd)
            write_eigen_vtk!(vtk_dir, pvd, 0, 0.0, xvi_h, xci_h, stokes, ez, Vx_v, Vy_v)
            last_vtk_t = 0.0
            println("ParaView: $(pvd).pvd   (time axis is seconds)")
            println("stations: $(joinpath(vtk_dir, "fault_stations.csv"))")
            flush(stdout)
        end
        while it < nsteps && t < max_time
            it += 1
            wall0 = time()
            out = eigenstrain_physical_step!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg, ez;
                inertia = true,
                ρ_inertia = ρ,
                solve_kwargs = (;
                    verbose_PH = false,
                    verbose_DR = false,
                    iterMax = 2.0e4,
                    iterMax_PH = 6,
                    nout = 50,
                    rel_drop = 1.0e-3,
                    linear_viscosity = true,
                    update_material = false,
                    # x = 0 is the free surface. The top wall is the plate at Vp.
                    boundary_vx = (; free_surface = :right, top = Vp),
                ),
            )
            apply_plate_velocity!(stokes, Vp)
            t += dt
            wall = time() - wall0
            τ_row = Array(stokes.τ.xy)[:, j_fault]
            V_row = Array(ez.V)[:, j_fault]
            ψ_row = Array(ez.ψ)[:, j_fault]
            all(isfinite, τ_row) && all(isfinite, V_row) && isfinite(out.rel) ||
                error("non-finite state at it=$it t=$t s")
            t_year = t / YEAR
            i125 = stations[3].i
            ref125 = isnothing(reference) ? nothing : baseline_at(reference[12.5e3], t)
            if isnothing(ref125)
                @printf(
                    "it=%d  t=%9.4f yr  passes=%d  rel=%.3e  Vmax=%.3e  dt=%.3e s  wall=%.2f s  12.5km τ=%.4f MPa V=%.3e\n",
                    it, t_year, out.passes, out.rel, out.V_max, dt, wall, τ_row[i125] / 1.0e6, V_row[i125],
                )
            else
                @printf(
                    "it=%d  t=%9.4f yr  passes=%d  rel=%.3e  Vmax=%.3e  dt=%.3e s  wall=%.2f s  12.5km τ=%.4f MPa ref=%.4f Δ=%+.4f  V=%.3e refV=%.3e\n",
                    it, t_year, out.passes, out.rel, out.V_max, dt, wall,
                    τ_row[i125] / 1.0e6, ref125.τ / 1.0e6, (τ_row[i125] - ref125.τ) / 1.0e6,
                    V_row[i125], ref125.V,
                )
                @printf("         ")
                for s in stations
                    ref = baseline_at(reference[s.depth], t)
                    @printf(
                        "%gkm τ=%.4f (ref %.4f)  ",
                        s.depth / 1.0e3, τ_row[s.i] / 1.0e6, ref.τ / 1.0e6,
                    )
                end
                @printf("\n")
            end
            flush(stdout)
            if !isnothing(hist)
                τ_ref = isnothing(ref125) ? NaN : ref125.τ
                V_ref = isnothing(ref125) ? NaN : ref125.V
                println(
                    hist,
                    "$it,$t_year,$t,$(out.passes),$(out.rel),$(out.V_max),$dt,$(τ_row[i125]),$τ_ref,$(V_row[i125]),$V_ref,$wall",
                )
                flush(hist)
            end
            write_stations!(probe, t, dt, τ_row, V_row, ψ_row)
            fast = out.V_max > 1.0e-4
            due = fast ? (t - last_vtk_t >= 1.0) : (t - last_vtk_t >= 0.05 * YEAR)
            if !isnothing(pvd) && (due || it == 1)
                write_eigen_vtk!(vtk_dir, pvd, it, t, xvi_h, xci_h, stokes, ez, Vx_v, Vy_v)
                last_vtk_t = t
            end
            dt = clamp(eigenstrain_lamem_dt(ez; ν = ν), 1.0e-6, 1.0e9)
            args = (; args..., dt = dt)
        end
        if !isnothing(pvd)
            write_eigen_vtk!(vtk_dir, pvd, it, t, xvi_h, xci_h, stokes, ez, Vx_v, Vy_v)
        end
        println("stopped at t=$(t / YEAR) yr after $it steps")
        flush(stdout)
    finally
        isnothing(hist) || close(hist)
        isnothing(probe) || close(probe)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    Wf = 100.0
    Wf_height = 40.0e3
    # 160 km × 160 km. 100 m cells across a 20-cell fault band and down to 20 km.
    # Ratio 3 toward the sides and below 20 km. Wf_height stays the 40 km RSF cutoff.
    mesh = seas_bp1_graded_mesh(;
        x0 = -160.0e3, x1 = 0.0, x_fine = -20.0e3, h_x = Wf,
        y0 = -80.0e3, y1 = 80.0e3, h_fault = Wf, n_fault = 20, ratio = 3.0,
    )
    println("ni = $(mesh.ni)   max adjacent ratio = $(round(mesh.jump; digits = 3))")
    println("dx [m], bottom → surface:")
    println(join(round.(mesh.dx; digits = 2), ", "))
    println("dy [m], −80 km → +80 km:")
    println(join(round.(mesh.dy; digits = 2), ", "))
    flush(stdout)
    nx, ny = mesh.ni
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    vtk_dir = joinpath(@__DIR__, "VTK_SEAS_BP1_FD_160km", "vtk")
    main(
        igg;
        xvi = mesh.xvi,
        Wf = Wf,
        Wf_height = Wf_height,
        nsteps = 20_000,
        max_time = 5.8e9,
        vtk_dir = vtk_dir,
        reference_dir = "/Users/Ibrag003/project/rsf/LaMEM_3/LaMEM_RSF/examples/BuiltInSetups/domain_search/baseline_series",
    )
end
