push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

@parallel_indices (i, j) function _init_single_phase_2D!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

"""Outer cells that fill `L`, growing at `ratio`, with the first jump below `jump_max`."""
function cell_sizes_ratio(L, h_fine; ratio = 3.0, jump_max = 3.5)
    h1 = ratio * h_fine
    best_n, best = 0, Inf
    n = 1
    while n ≤ 60
        s = h1 * (ratio^n - 1) / (ratio - 1)
        h_first = h1 * L / s
        r = max(h_first / h_fine, h_fine / h_first)
        if r < jump_max
            err = abs(log(s / L))
            err < best && (best = err; best_n = n)
        end
        s > L * jump_max && best_n > 0 && break
        n += 1
    end
    best_n > 0 || error("cannot fill $L m from $h_fine m at ratio $ratio")
    scale = L / (h1 * (ratio^best_n - 1) / (ratio - 1))
    return [h1 * ratio^(k - 1) * scale for k in 1:best_n]
end

"""
Graded mesh with the coarse cells on the free-surface column.

The uniform 4×40 strip is periodic, so every Vy row receives `ρ/dt`. This mesh
is not. The free-surface column is the last Vy row, and it is coarse enough that
`ρ/dt` is the whole diagonal at the coseismic step (`dt ≈ 0.016 s`). That is the
row the 160 km run skipped, taken where missing the mass overflows in one solve.
"""
function coarse_surface_vertices(; h = 100.0, n_fine_x = 6, L_coarse_x = 12.0e3, n_fault = 4, L_side = 6.0e3)
    xv = Float64[0.0]
    x = 0.0
    for _ in 1:n_fine_x
        x += h
        push!(xv, x)
    end
    for hc in cell_sizes_ratio(L_coarse_x, h)
        x += hc
        push!(xv, x)
    end
    y0 = -(n_fault * h / 2 + L_side)
    yv = Float64[y0]
    y = y0
    for hc in reverse(cell_sizes_ratio(L_side, h))
        y += hc
        push!(yv, y)
    end
    for _ in 1:n_fault
        y += h
        push!(yv, y)
    end
    for hc in cell_sizes_ratio(L_side, h)
        y += hc
        push!(yv, y)
    end
    return xv, yv
end

@testset "coseismic eigenstrain on a graded free surface" begin
    # State of the 160 km GPU blow-up: τ still ~35 MPa, V ~ 0.1 m/s, dt ~ 0.016 s.
    dt = 1.574e-2
    τ = 34.6e6
    Vslip = 0.109
    ρ = 2670.0
    G, η, ν = 3.2038e10, 5.0e26, 0.25
    Wf, σ = 100.0, 50.0e6
    a, b, μ₀, V₀, L, Vp = 0.01, 0.015, 0.6, 1.0e-6, 0.008, 1.0e-9

    xv, yv = coarse_surface_vertices()
    dx = diff(xv)
    @test maximum(dx) / Wf > 50
    @test maximum(dx[2:end] ./ dx[1:(end - 1)]) ≤ 3.5

    nx, ny = length(xv) - 1, length(yv) - 1
    ni = (nx, ny)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    grid = Geometry(PTArray(backend), xv, yv)
    yv_h = Array(grid.xvi[2])
    j_expect = argmin(abs.(yv_h))
    @test yv_h[j_expect] ≈ 0.0
    @test yv_h[j_expect] - yv_h[j_expect - 1] ≈ Wf
    @test yv_h[j_expect + 1] - yv_h[j_expect] ≈ Wf

    el = ConstantElasticity(; G = G, ν = ν)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η), el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    proto = RateStateFriction(0.0, μ₀, V₀, a, b, L, 0.0, Wf)
    ez = eigenstrain_rsf(backend, ni; Wf = Wf, G = G, σ = σ, law = proto, tol = 1.0e-2, max_pass = 8)
    j_fault = init_eigenstrain_row!(
        ez, grid.xvi; y_fault = 0.0, a = a, b = b, ψ = 0.0, V_lag = Vslip,
    )
    @test j_fault == j_expect
    set_eigenstrain_ψ_from_stress!(ez, τ, Vslip)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = false, right = false, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, τ)
    fill!(stokes.τ_o.xy, τ)
    yVx = Array(grid.xi_vel[1][2])
    Vx = [y >= 0.0 ? Vp : 0.0 for _ in xv, y in yVx]
    Vx[:, end] .= 2 * Vp .- Vx[:, end - 1]
    copyto!(stokes.V.Vx, Vx)
    fill!(stokes.V.Vy, 0.0)
    stokes.V0.Vx .= stokes.V.Vx
    stokes.V0.Vy .= stokes.V.Vy
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5, CFL = 0.9)

    out = eigenstrain_physical_step!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg, ez;
        inertia = true,
        ρ_inertia = ρ,
        solve_kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 5.0e2,
            iterMax_PH = 4,
            nout = 50,
            rel_drop = 1.0e-3,
            linear_viscosity = true,
            update_material = false,
            boundary_vx = (; free_surface = :right, top = Vp),
        ),
    )

    @test out.passes ≥ 1
    @test isfinite(out.rel)
    @test isfinite(out.V_max)
    @test all(isfinite, Array(stokes.τ.xy))
    @test all(isfinite, Array(stokes.τ.xx))
    @test all(isfinite, Array(stokes.τ.yy))
    @test all(isfinite, Array(stokes.V.Vx))
    @test all(isfinite, Array(stokes.V.Vy))
    @test all(isfinite, Array(ez.V))
    @test all(>=(0), Array(ez.V))
    # Every momentum row, including the free-surface Vy column, carries ρ/dt.
    m = ρ / dt
    Dx, Dy = Array(dyrel.Dx), Array(dyrel.Dy)
    @test minimum(Dx) ≥ m * (1 - 1.0e-12)
    @test minimum(Dy) ≥ m * (1 - 1.0e-12)
    @test minimum(Dy[end, :]) ≥ m * (1 - 1.0e-12)
    @test all(isfinite, Array(dyrel.λmaxVx))
    @test all(isfinite, Array(dyrel.λmaxVy))
    @test all(isfinite, Array(dyrel.cVx))
    @test all(isfinite, Array(dyrel.cVy))
end
