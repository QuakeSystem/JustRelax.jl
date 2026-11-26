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

@parallel_indices (i, j) function _fill_fault_phases!(phases, j_lo, j_hi)
    if j_lo ≤ j ≤ j_hi
        @index phases[1, i, j] = 0.0
        @index phases[2, i, j] = 1.0
    else
        @index phases[1, i, j] = 1.0
        @index phases[2, i, j] = 0.0
    end
    return nothing
end

@testset "DYREL 2D GeoParams RSF" begin
    init_mpi = !JustRelax.MPI.Initialized()
    nx, ny = 16, 16
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    ni = (nx, ny)
    li = (1.0e3, 1.0e3)
    origin = (0.0, 0.0)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid
    dt = 500.0
    V_top = 4.0e-9
    P0 = 5.0e6

    η0 = 1.0e23
    G0 = 3.0e10
    el = ConstantElasticity(; G = G0, ν = 0.25)
    visc = LinearViscous(; η = η0)
    # Phase 2 only. Ω = 1 so healing V₀ dt / L is visible; Ω = 40 would not move.
    rsf = GeoParams.RateStateFriction(0.0, 0.2, V_top, 0.011, 0.017, 0.01, 0.0, 50.0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, rsf, el)),
            Elasticity = el,
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    j_lo, j_hi = ny ÷ 2 - 1, ny ÷ 2 + 1
    @parallel (@idx ni) _fill_fault_phases!(phase_ratios.center, j_lo, j_hi)
    @parallel (@idx ni .+ 1) _fill_fault_phases!(phase_ratios.vertex, j_lo, j_hi)

    fields = RateStateArrays(backend, ni; loc = :vertex)
    init_geoparams_rate_state!(
        fields, rheology, phase_ratios, (0.0, 1.0), xvi; xci = xci,
    )
    @test maximum(Array(fields.rsf_mask_v)) ≈ 1.0
    @test maximum(Array(fields.rsf_mask)) == 0.0
    Ωv_before = copy(Array(fields.Ω_old_v))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, P0)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (1.0e10, 1.0e28))
    η_before = copy(Array(stokes.viscosity.η))

    yVx = grid.xi_vel[1][2]
    ybot, ytop = origin[2], origin[2] + li[2]
    stokes.V.Vx .= PTArray(backend)([
            V_top * (y - ybot) / (ytop - ybot) for _ in xvi[1], y in yVx
        ])
    fill!(stokes.V.Vy, 0.0)
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
    @views stokes.V.Vx[:, end] .= V_top
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-5)
    out = solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 20.0e3,
            nout = 50,
            rel_drop = 1.0e-2,
            linear_viscosity = true,
            viscosity_cutoff = (1.0e10, 1.0e28),
            gp_rsf = fields,
        ),
    )

    @test isfinite(out.dt_rsf) && out.dt_rsf > 0
    @test all(isfinite, Array(stokes.τ.xy))
    @test all(isfinite, Array(stokes.viscosity.ηv))
    # Centers are not an RSF location, so the post-stress refresh must not touch η.
    @test Array(stokes.viscosity.η) ≈ η_before
    Ωv_after = Array(fields.Ω_old_v)
    @test any(abs.(Ωv_after .- Ωv_before) .> 0)
    @test all(isfinite, Ωv_after)

    εbg = V_top / li[2]
    τ_elastic = εbg * G0 * dt
    @test isapprox(maximum(abs, Array(stokes.τ.xy)), τ_elastic; rtol = 0.15)
    @test out.Vp_max < 1.0e-3 * V_top
    @test minimum(Array(stokes.viscosity.η)) > 1.0e22
    @test minimum(Array(stokes.viscosity.ηv)) > 1.0e22

    finalize_global_grid(; finalize_MPI = false)
end
