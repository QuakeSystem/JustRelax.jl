push!(LOAD_PATH, "..")

const _backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")

@static if _backend_name == "AMDGPU"
    using AMDGPU
elseif _backend_name == "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if _backend_name == "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustRelax.AMDGPUBackend
elseif _backend_name == "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if _backend_name == "AMDGPU"
    AMDGPU.ROCBackend
elseif _backend_name == "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

@parallel_indices (i, j, k) function _fill_phase1_bp1!(phases)
    @index phases[1, i, j, k] = 1.0
    return nothing
end

@testset "SEAS BP1 prestress" begin
    σ, μ0, b, V0, V = 50.0e6, 0.6, 0.015, 1.0e-6, 1.0e-9
    τ0 = seas_bp1_tau0(0.025, σ, μ0, b, V0, V)
    @test isfinite(τ0) && τ0 > 0
    Ω_ss = seas_bp1_omega(0.025, b, μ0, τ0, σ, V0, V)
    @test Ω_ss ≈ log(V0 / V) rtol = 1.0e-10
    Ω_shallow = seas_bp1_omega(0.010, b, μ0, τ0, σ, V0, V)
    r = GeoParams.RateStateFriction(0.0, μ0, V0, 0.010, b, 0.008, 0.0, 100.0)
    ε = V / (2 * 100.0)
    τ = compute_τII(r, ε; P = σ, Ω_old = Ω_shallow, a = 0.010, b = b)
    @test τ ≈ τ0 rtol = 1.0e-8
end

@testset "DYREL 3D antiplane GeoParams RSF" begin
    init_mpi = !JustRelax.MPI.Initialized()
    nx, ny, nz = 2, 8, 5
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_mpi)...)
    xv = collect(range(-50.0, 50.0; length = nx + 1))
    yv = collect(range(-400.0, 400.0; length = ny + 1))
    zv = collect(range(-500.0, 0.0; length = nz + 1))
    grid = Geometry(PTArray(backend), xv, yv, zv)
    ni = grid.ni
    σ, Vp, D, dt = 50.0e6, 1.0e-9, 100.0, 2522.0
    el = ConstantElasticity(; G = 3.2038e10, ν = 0.25)
    visc = LinearViscous(; η = 5.0e26)
    rsf = GeoParams.RateStateFriction(0.0, 0.6, 1.0e-6, 0.010, 0.015, 0.008, 0.0, D)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2670.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, rsf, el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    for loc in (phase_ratios.center, phase_ratios.vertex, phase_ratios.xy, phase_ratios.xz, phase_ratios.yz, phase_ratios.Vx, phase_ratios.Vy, phase_ratios.Vz)
        @parallel (@idx size(loc)) _fill_phase1_bp1!(loc)
    end
    fields = RateStateXY(ni; skip_vs = true)
    meta = init_antiplane_fault!(
        fields, grid.xvi, grid.xci;
        z_rsf_min = -300.0, V_plate = Vp, σ = σ,
    )
    mask = Array(fields.rsf_mask)
    zc = Array(grid.xci[3])
    j = meta.j_fault
    @test yv[j] == 0.0
    @test count(>(0.5), mask) == size(mask, 1) * count(z -> z >= -300.0, zc)
    @test all(mask[:, j, zc .< -300.0] .== 0.0)
    @test all(mask[:, 1, :] .== 0.0)

    flow_bcs = VelocityBoundaryConditions(;
        no_slip = (left = false, right = false, front = false, back = false, bot = false, top = false),
        free_slip = (left = false, right = false, front = false, back = false, bot = true, top = true),
        periodic = (left = true, right = true, front = false, back = false, bot = false, top = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, σ)
    fill!(stokes.τ.xy, meta.τ0)
    fill!(stokes.τ_o.xy, meta.τ0)
    Vx = zeros(size(stokes.V.Vx))
    for jv in 1:size(Vx, 2)
        Vx[:, jv, :] .= jv <= j ? -meta.half : meta.half
    end
    copyto!(stokes.V.Vx, Vx)
    copyto!(stokes.V0.Vx, Vx)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    fixed_before = copy(Array(stokes.V.Vx))

    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    cutoff = (1.0, 5.0e26)
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff)
    apply_geoparams_rsf_viscosity!(stokes, fields, phase_ratios, rheology, dt; viscosity_cutoff = cutoff)
    ηe = Array(fields.η)
    @test all(isfinite, ηe[mask .> 0.5])
    @test all(>(0), ηe[mask .> 0.5])

    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-4)
    out = solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 30.0,
            nout = 10,
            rel_drop = 1.0e-2,
            linear_viscosity = true,
            viscosity_cutoff = cutoff,
            gp_rsf = fields,
            inertia = true,
            ρ_inertia = 2670.0,
        ),
    )
    Vx_after = Array(stokes.V.Vx)
    fixed = Array(fields.vx_fixed)
    @test all(isfinite, Array(stokes.τ.xy))
    @test all(isfinite, Array(fields.Ω))
    @test isfinite(out.Vp_max)
    @test Vx_after[fixed .> 0.5] ≈ fixed_before[fixed .> 0.5] rtol = 1.0e-10
    finalize_global_grid(; finalize_MPI = false)
end
