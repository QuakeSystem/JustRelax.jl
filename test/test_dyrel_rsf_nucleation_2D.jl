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

@parallel_indices (i, j) function _fill_nucleation_fault!(phases, j_fault)
    if j == j_fault
        @index phases[1, i, j] = 0.0
        @index phases[2, i, j] = 1.0
    else
        @index phases[1, i, j] = 1.0
        @index phases[2, i, j] = 0.0
    end
    return nothing
end

# Fault row of the viscosity Herrendörfer run at the last healthy step (it = 1204),
# x = -55 km to -39.5 km. The next accepted step used dt = 0.5163 s, did not converge,
# and returned 9.7 MPa with the slip rate on the 100 m/s cap.
const _τ_row = [
    1.073413e6, 1.07798925e6, 1.08300112e6, 1.08860525e6, 1.09509138e6,
    1.10304375e6, 1.11371188e6, 1.12984662e6, 1.15754288e6, 1.21014212e6,
    1.31526588e6, 1.51420038e6, 1.70575288e6, 7.73298096e2, 9.34234741e2,
    1.387799e6, 7.57788025e2, 9.03236267e2, 7.34492493e2, 1.23019325e6,
    1.48880444e3, 1.14745162e6, 6.62938171e2, 7.15481201e2, 5.59955139e2,
    6.59840881e2, 1.71986462e3, 1.0061415e4, 9.96385812e5, 7.3826425e5,
    7.38601938e5, 5.3587925e5,
]
const _Ω_row = [
    3.70009005e-1, 3.45312893e-1, 3.18032384e-1, 2.87783891e-1, 2.54130065e-1,
    2.16575131e-1, 1.7456001e-1, 1.27457008e-1, 7.45631233e-2, 1.50904283e-2,
    -5.18582612e-2, -1.27934039e-1, -2.35154405e-1, -3.09465051e-1, -4.85417366e-1,
    -6.5471065e-1, -8.48410606e-1, -9.39637899e-1, -1.18258643e0, -1.46353447e0,
    -1.58578968e0, -1.88584495e0, -2.22251439e0, -2.44015217e0, -2.79882216e0,
    -3.17494559e0, -3.41154814e0, -3.61991739e0, -4.05349159e0, -4.5433917e0,
    -5.34070253e0, -6.69148064e0,
]
const _Vy_row = [
    1.117822e-4, 1.478976e-4, 2.013099e-4, 2.803908e-4, 3.989137e-4, 5.814907e-4,
    8.7509e-4, 1.371144e-3, 2.238122e-3, 3.72265e-3, 5.827796e-3, 5.862133e-3,
    -1.760168e-2, -2.68883e-2, 1.187699e-3, -1.517203e-2, -3.972295e-2, -2.974071e-2,
    -1.93325e-2, -4.078824e-2, -4.065148e-2, -3.635653e-2, -4.77387e-2, -3.04488e-2,
    -2.819205e-2, -2.354984e-2, -1.731421e-2, -5.871286e-3, -1.035875e-2, -1.520979e-2,
    -1.354222e-2, -1.290262e-2,
]

@testset "DYREL viscosity RSF nucleation step" begin
    nx = length(_τ_row)
    h = 500.0
    x0 = -55.0e3
    # Same fault-normal jump as the run: 500 m on the fault, then several kilometres.
    yv = [-90.8e3, -76.0e3, -75.5e3, -75.0e3, -74.5e3, -74.0e3, -66.6e3]
    xv = [x0 + (i - 1) * h for i in 1:(nx + 1)]
    ny = length(yv) - 1
    ni = (nx, ny)
    dt = 0.5162998092795753
    P0 = 5.0e6
    G0 = 3.0e10
    V_plate = 4.0e-9

    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)
    grid = Geometry(PTArray(backend), xv, yv)
    (; xci, xvi) = grid

    el = ConstantElasticity(; G = G0, ν = 0.25)
    visc = LinearViscous(; η = 5.0e26)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2700.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 2700.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    j_fault = 4
    @parallel (@idx ni) _fill_nucleation_fault!(phase_ratios.center, j_fault)
    @parallel (@idx ni .+ 1) _fill_nucleation_fault!(phase_ratios.vertex, j_fault)

    RSF = (
        enabled = true,
        loc = :vertex,
        V0 = V_plate,
        dt_min = 0.1,
        dt_max = 1.0e7,
        dt_rsf_switch = 1.0e9,
        G = G0,
        ν = 0.25,
        affect_stokes = true,
        Phase1 = (
            a_rsf = 0.011, b_rsf = 0.017, mu0_rsf = 0.2, D_rs = 0.01, Wf = 500.0,
            state_rsf_init = 40.0, λ = 0.0, C = 0.0,
        ),
        Phase2 = (
            a_rsf = 0.011,
            b_rsf = 0.001,
            b_rsf_val = (0.001, 0.017, 0.017, 0.001),
            b_rsf_x = (-47000.0, -43000.0, 33000.0, 37000.0),
            mu0_rsf = 0.2, D_rs = 0.01, Wf = 500.0,
            state_rsf_init = -1.0, λ = 0.0, C = 0.0,
        ),
    )
    ctrl = build_rate_state_controller(RSF; nphases = 2, di = (h, h))
    fields = RateStateArrays(backend, ni; loc = :vertex)
    init_rate_state_fields!(fields, ctrl, phase_ratios, xci, xvi)
    Ωv = Array(fields.Ωv)
    Ωv[:, j_fault] = vcat(_Ω_row, _Ω_row[1])
    copyto!(fields.Ωv, Ωv)
    copyto!(fields.Ω_old_v, Ωv)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    fill!(stokes.P, P0)
    for τ in (stokes.τ, stokes.τ_o)
        xy = fill(1.06e6, size(Array(τ.xy)))
        xy[:, j_fault] = vcat(_τ_row, _τ_row[1])
        copyto!(τ.xy, xy)
        fill!(τ.xx, 0.0)
        fill!(τ.yy, 0.0)
    end
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    cutoff = (1.0e6, 5.0e26)
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff)

    yVx = Array(grid.xi_vel[1][2])
    ybot, ytop = grid.origin[2], grid.origin[2] + grid.li[2]
    stokes.V.Vx .= PTArray(backend)(
        [
            V_plate * (y - ybot) / (ytop - ybot) for _ in Array(xvi[1]), y in yVx
        ]
    )
    fill!(stokes.V.Vy, 0.0)
    vy = Array(stokes.V.Vy)
    jv = min(j_fault, size(vy, 2))
    nset = min(length(_Vy_row), size(vy, 1))
    vy[1:nset, jv] .= _Vy_row[1:nset]
    size(vy, 1) > length(_Vy_row) && (vy[length(_Vy_row) + 1, jv] = _Vy_row[1])
    copyto!(stokes.V.Vy, vy)
    @views stokes.V.Vx[:, end] .= V_plate
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    copyto!(stokes.V0.Vx, stokes.V.Vx)
    copyto!(stokes.V0.Vy, stokes.V.Vy)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)
    out = solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 5.0e3,
            nout = 10,
            rel_drop = 1.0e-4,
            linear_viscosity = true,
            viscosity_cutoff = cutoff,
            rsf = (; ctrl = ctrl, fields = fields),
            inertia = true,
            ρ_inertia = 2700.0,
        ),
    )

    τ = Array(stokes.τ.xy)
    Vp = Array(fields.Vp_v)
    Ω = Array(fields.Ωv)
    @test all(isfinite, τ)
    @test all(isfinite, Vp)
    @test all(isfinite, Ω)
    @test all(isfinite, Array(stokes.V.Vx))
    @test all(isfinite, Array(stokes.V.Vy))
    # The uncapped solve returned ~9.6 MPa, Vp = 100 m/s and Ω ≈ -24, then the
    # iteration count sat on the 5e4 cap.
    @test out.iter < 20_000
    @test maximum(abs, τ) < 3.0e6
    @test maximum(Vp) < 1.0
    @test minimum(Ω[1:(end - 1), j_fault]) > -10.0
    @test out.dt_rsf > 1.0e-3

    finalize_global_grid(; finalize_MPI = false)
end

@parallel_indices (i, j) function _fill_shear_jump_band!(phases, j0, j1)
    if j0 ≤ j ≤ j1
        @index phases[1, i, j] = 0.0
        @index phases[2, i, j] = 1.0
    else
        @index phases[1, i, j] = 1.0
        @index phases[2, i, j] = 0.0
    end
    return nothing
end

@testset "DYREL viscosity RSF shear jump on a graded fault" begin
    # The viscosity run at it = 1250 was still at 2.1 MPa, but the velocity already
    # jumped by ~0.13 m/s across the fault, antisymmetric in y. The inner solve is
    # cut at `iterMax` (1000 in that run; 100 here) before that jump has relaxed.
    # Folding a new η from the unfinished velocity, and clearing the search
    # direction, leaves the residual above 1 until the iteration budget runs out.
    # One frozen η converges. A rejected step would come back as dt/2 = 0.179 s.
    nx = 16
    yv = [
        -150.0e3, -142.6e3, -135.2e3, -127.8e3, -120.4e3, -113.0e3, -105.6e3,
        -98.2e3, -90.8e3, -83.4e3, -76.0e3, -75.5e3, -75.0e3, -74.5e3, -74.0e3,
        -66.6e3, -59.2e3, -51.8e3, -44.4e3, -37.0e3, -29.6e3, -22.2e3, -14.8e3,
        -7.4e3, 0.0,
    ]
    h = 500.0
    xv = [-55.0e3 + (i - 1) * h for i in 1:(nx + 1)]
    ny = length(yv) - 1
    dt = 0.357619
    P0 = 5.0e6
    G0 = 3.0e10
    V_plate = 4.0e-9
    j_fault = findfirst(==(-75.0e3), yv)

    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)
    grid = Geometry(PTArray(backend), xv, yv)
    (; xci, xvi) = grid

    el = ConstantElasticity(; G = G0, ν = 0.25)
    visc = LinearViscous(; η = 5.0e26)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2700.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 2700.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el)),
            Elasticity = el,
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, 2, (nx, ny))
    @parallel (@idx (nx, ny)) _fill_shear_jump_band!(phase_ratios.center, j_fault - 1, j_fault)
    @parallel (@idx (nx, ny) .+ 1) _fill_shear_jump_band!(
        phase_ratios.vertex, j_fault - 1, j_fault + 1
    )

    RSF = (
        enabled = true,
        loc = :vertex,
        V0 = V_plate,
        dt_min = 1.0e-2,
        dt_max = 1.0e7,
        dt_rsf_switch = 1.0e9,
        G = G0,
        ν = 0.25,
        affect_stokes = true,
        Phase1 = (
            a_rsf = 0.011, b_rsf = 0.017, mu0_rsf = 0.2, D_rs = 0.01, Wf = 500.0,
            state_rsf_init = 40.0, λ = 0.0, C = 0.0,
        ),
        Phase2 = (
            a_rsf = 0.011, b_rsf = 0.017, mu0_rsf = 0.2, D_rs = 0.01, Wf = 500.0,
            state_rsf_init = -1.0, λ = 0.0, C = 0.0,
        ),
    )
    ctrl = build_rate_state_controller(RSF; nphases = 2, di = (h, h))
    fields = RateStateArrays(backend, (nx, ny); loc = :vertex)
    init_rate_state_fields!(fields, ctrl, phase_ratios, xci, xvi)
    Ωv = Array(fields.Ωv)
    Ωv[:, j_fault] .= -0.45
    copyto!(fields.Ωv, Ωv)
    copyto!(fields.Ω_old_v, Ωv)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, (nx, ny), flow_bcs)
    fill!(stokes.P, P0)
    for τ in (stokes.τ, stokes.τ_o)
        xy = fill(1.0e6, size(Array(τ.xy)))
        xy[:, j_fault] .= 1.72e6
        copyto!(τ.xy, xy)
        fill!(τ.xx, 0.0)
        fill!(τ.yy, 0.0)
    end
    ρg = @zeros((nx, ny)...), @zeros((nx, ny)...)
    args = (; T = @zeros((nx, ny) .+ 2...), P = stokes.P, dt = dt)
    cutoff = (1.0e6, 5.0e26)
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff)

    yVx = Array(grid.xi_vel[1][2])
    vx = [y < -75.0e3 ? -0.13 : 0.13 for _ in xv, y in yVx]
    vx[:, end] .= V_plate
    stokes.V.Vx .= PTArray(backend)(vx)
    fill!(stokes.V.Vy, 0.0)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    copyto!(stokes.V0.Vx, stokes.V.Vx)
    copyto!(stokes.V0.Vy, stokes.V.Vy)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)
    out = solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs = (;
            verbose_PH = false,
            verbose_DR = false,
            iterMax = 100.0,
            total_iterMax = 2.0e3,
            nout = 10,
            rel_drop = 1.0e-4,
            linear_viscosity = true,
            viscosity_cutoff = cutoff,
            rsf = (; ctrl = ctrl, fields = fields),
            inertia = true,
            ρ_inertia = 2700.0,
        ),
    )

    τ = Array(stokes.τ.xy)
    Vp = Array(fields.Vp_v)
    @test all(isfinite, τ)
    @test all(isfinite, Vp)
    # Refolding η on every outer pass stays above the tolerance and returns at
    # the 2000-iteration cap, with dt_rsf = dt/2.
    @test out.iter < 1_000
    @test maximum(abs, τ) < 3.0e6
    @test maximum(Vp) < 10.0
    @test 0.0 < out.dt_rsf < 1.0e-2

    finalize_global_grid(; finalize_MPI = false)
end
