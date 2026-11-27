push!(LOAD_PATH, "..")

const JR_BACKEND = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if JR_BACKEND === "AMDGPU"
    using AMDGPU
elseif JR_BACKEND === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if JR_BACKEND === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif JR_BACKEND === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if JR_BACKEND === "AMDGPU"
    AMDGPU.ROCBackend
elseif JR_BACKEND === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

# Dense block (phase 2) in a unit box: x ∈ [0.45, 0.75], y ∈ [0.35, 0.65].
@parallel_indices (i, j) function init_block!(phases, xc, yc)
    inside = abs(xc[i] - 0.6) ≤ 0.15 && abs(yc[j] - 0.5) ≤ 0.15
    @index phases[1, i, j] = inside ? 0.0 : 1.0
    @index phases[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

# Velocity box: the (Vx, Vy) nodes within `half` of `center` are prescribed to `v`.
function velocity_box_bcs(stokes, grid; center, half, v, components = (:Vx, :Vy))
    dirichlet = map(components) do c
        d = c === :Vx ? 1 : 2
        x, y = map(Array, grid.xi_vel[d])
        V = getproperty(stokes.V, c)
        mask = [Float64(abs(x[i] - center[1]) ≤ half[1] && abs(y[j] - center[2]) ≤ half[2]) for i in axes(V, 1), j in axes(V, 2)]
        value = mask .* v[d]
        c => JustRelax.DirichletBoundaryCondition(
            copyto!(similar(V), value), JustRelax.Mask(copyto!(similar(V), mask)),
        )
    end
    return VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        dirichlet = (; dirichlet...),
    )
end

const BOX = (; center = (0.25, 0.5), half = (0.08, 0.12), v = (0.3, -0.2))

observed_sum(field, mask) =
    sum(Array(field)[first(mask.i):last(mask.i), first(mask.j):last(mask.j)])

# Returns J, the adjoint dJ/d`parameter` of the block (phase 2) and the final state.
# `solver = :variational` runs `solve_VariationalDYREL!` with ϕ ≡ 1, `:standard` `solve_DYREL!`.
# `n` is the power-law exponent of the matrix (1: linear viscosity).
function sinking_block_box(
        igg; observation, adjoint = false, parameter = :ρ, box = true,
        ρ_block = 1.2, η_block = 10.0, adjoint_kwargs = (;), solver = :variational, n = 1,
    )
    ni = (24, 24)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    elasticity = ConstantElasticity(; G = 1.0e6, Kb = 2.0)
    material(phase, ρ, creep) = SetMaterialParams(;
        Phase = phase,
        Density = ConstantDensity(; ρ),
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology((creep, elasticity)),
        Gravity = ConstantGravity(; g = 1.0),
    )
    matrix_creep = isone(n) ? LinearViscous(; η = 1.0) : PowerlawViscous(; η0 = 1.0, n, ε0 = 1.0)
    rheology = (material(1, 1.0, matrix_creep), material(2, ρ_block, LinearViscous(; η = η_block)))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_block!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_block!(phase_ratios.vertex, grid.xvi...)

    stokes = StokesArrays(backend, ni)
    stokes_ad = AdjointStokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    dt = 1.0
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    flow_bcs = box ? velocity_box_bcs(stokes, grid; BOX...) :
        VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
    flow_bcs!(stokes, flow_bcs)

    gradients = material_controls(backend, ni, (parameter,); nphases = length(rheology))
    options = (;
        verbose_PH = false, verbose_DR = false, iterMax = 2.0e5, total_iterMax = 2.0e5,
        nout = 50, linear_viscosity = isone(n), adjoint, observation, gradients, adjoint_kwargs,
    )
    out = if solver === :variational
        ϕ = RockRatio(backend, ni)
        foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-10)
        solve_VariationalDYREL!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid,
            dt, igg; options...,
        )
    else
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-10)
        solve_DYREL!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid,
            dt, igg; options...,
        )
    end
    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    field = getproperty(stokes.V, observation.field)
    return (;
        J = observed_sum(field, mask),
        gradient = sum(Array(gradients[parameter].center)[2, :, :]),
        stokes, stokes_ad, flow_bcs, out,
    )
end

central_difference(f, x, h) = (f(x + h) - f(x - h)) / (2h)

@testset "Velocity Dirichlet (velocity box) forward + adjoint" begin
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(24, 24, 1; init_MPI = init_mpi)...)

    # observation above the block, away from the velocity box
    obs(field) = (; field, center = (0.6, 0.8), half_width = (0.1, 0.08))

    for solver in (:variational, :standard)
    @testset "$solver" begin
    solve_box(; kwargs...) = sinking_block_box(igg; solver, kwargs...)

    @testset "forward holds the prescribed velocity" begin
        r = solve_box(; observation = obs(:Vy))
        for (c, d) in ((:Vx, 1), (:Vy, 2))
            bc = getproperty(r.flow_bcs.dirichlet, c)
            m = Array(bc.mask.mask) .!= 0
            V = Array(getproperty(r.stokes.V, c))
            @test any(m)
            @test all(V[m] .== BOX.v[d])
        end
        @test r.out.converged
    end

    for field in (:Vx, :Vy), inner_solver in (:bicgstab, :dr)
        @testset "dJ/dρ_block, $field, $inner_solver" begin
            r = solve_box(; observation = obs(field), adjoint = true,
                adjoint_kwargs = (; inner_solver))
            fd = central_difference(ρ -> solve_box(; observation = obs(field), ρ_block = ρ).J, 1.2, 1.0e-4)
            @info "$solver dJ/dρ_block" field inner_solver adjoint = r.gradient fd rel = abs(r.gradient - fd) / abs(fd)
            @test r.gradient ≈ fd rtol = 1.0e-6
            # the momentum adjoint of a prescribed node is not an unknown
            for (c, λ) in ((:Vx, r.stokes_ad.λV.Vx), (:Vy, r.stokes_ad.λV.Vy))
                m = Array(getproperty(r.flow_bcs.dirichlet, c).mask.mask) .!= 0
                @test all(iszero, Array(λ)[m])
            end
        end
    end

    @testset "dJ/dη_block, Vy" begin
        r = solve_box(; observation = obs(:Vy), adjoint = true, parameter = :η)
        fd = central_difference(η -> solve_box(; observation = obs(:Vy), η_block = η).J, 10.0, 1.0e-3)
        @info "$solver dJ/dη_block" adjoint = r.gradient fd rel = abs(r.gradient - fd) / abs(fd)
        @test r.gradient ≈ fd rtol = 1.0e-6
    end

    @testset "the box changes the gradient" begin
        with_box = solve_box(; observation = obs(:Vy), adjoint = true).gradient
        without = solve_box(; observation = obs(:Vy), adjoint = true, box = false).gradient
        @info "$solver box vs no box" with_box without
        @test !isapprox(with_box, without; rtol = 1.0e-2)
    end

    @testset "observation on prescribed nodes is rejected" begin
        on_box = (; field = :Vx, center = BOX.center, half_width = (0.05, 0.05))
        @test_throws ArgumentError solve_box(; observation = on_box, adjoint = true)
    end

    @testset "dJ/dρ_block with a power-law matrix (nonlinear viscosity), Vy" begin
        r = solve_box(; observation = obs(:Vy), adjoint = true, n = 3)
        fd = central_difference(ρ -> solve_box(; observation = obs(:Vy), ρ_block = ρ, n = 3).J, 1.2, 1.0e-4)
        @info "$solver dJ/dρ_block, power law" adjoint = r.gradient fd rel = abs(r.gradient - fd) / abs(fd)
        @test r.gradient ≈ fd rtol = 1.0e-5
    end
    end
    end
end
