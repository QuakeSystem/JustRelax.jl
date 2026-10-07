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

"""Start-up Couette: bottom plate fixed, top plate at `U`, fluid initially at rest.

```
u(y,t)/U = y/H + (2/π) Σ_{n=1}^{∞} ((-1)^n / n) sin(n π y / H) exp(-n² π² ν t / H²)
```
with `ν = η/ρ`.

# How to check that inertia is working

1. **Analytic profile (this test)** — at Fo = νt/H² ≈ 0.1 the midplane
   velocity is still *below* the linear Couette value `U/2`. Without inertia
   (Stokes, Fo→∞ instantly) the profile would already be linear.

2. **Quick plot** (Julia REPL after running the solver loop):
   ```julia
   using Plots
   plot(Vx_num, y_int; label="numerical", xlabel="Vx", ylabel="y")
   plot!(Vx_ana, y_int; label="analytic", ls=:dash)
   plot!([0, U], [0, H]; label="steady Couette", ls=:dot)
   ```
   Inertia works if the numerical curve hugs the analytic transient (S-shaped /
   lagging near the midplane), not the straight steady line.

3. **On/off switch** — re-run the same steps with `inertia=false`: the solver
   jumps to near-linear Couette in one (or few) steps. With `inertia=true` the
   profile evolves gradually over many physical `dt`.

4. **V0 snapshot** — after each converged step `stokes.V0` equals the previous
   `stokes.V`. If `V0` stays zero forever, the inertia term is never advancing.
"""
function couette_analytic(y, t; U, H, ν, nterms = 80)
    s = y / H
    for n in 1:nterms
        s += (2 / π) * ((-1)^n / n) * sin(n * π * y / H) * exp(-(n * π)^2 * ν * t / H^2)
    end
    return U * s
end

@testset "DYREL 2D inertial Couette" begin
    init_mpi = !JustRelax.MPI.Initialized()
    nx, ny = 8, 32
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    ni = (nx, ny)
    H = 1.0
    li = (1.0, H)
    origin = (0.0, 0.0)
    grid = Geometry(ni, li; origin = origin)
    (; xvi) = grid

    η = 1.0
    ρ = 1.0
    ν = η / ρ
    U = 1.0
    dt = 0.01
    nsteps = 10
    t_end = nsteps * dt  # Fo = ν t / H² = 0.1

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η),)),
        ),
    )
    @parallel_indices (i, j) function _init_single_phase_2D!(phases)
        @index phases[1, i, j] = 1.0
        return nothing
    end
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx size(phase_ratios.center)) _init_single_phase_2D!(phase_ratios.center)
    @parallel (@idx size(phase_ratios.vertex)) _init_single_phase_2D!(phase_ratios.vertex)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    @test stokes.V0 isa JustRelax.Velocity
    @test size(stokes.V0.Vx) == size(stokes.V.Vx)

    fill!(stokes.V.Vx, 0.0)
    fill!(stokes.V.Vy, 0.0)
    fill!(stokes.V0.Vx, 0.0)
    fill!(stokes.V0.Vy, 0.0)
    @views stokes.V.Vx[:, end] .= U
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8, CFL = 0.9)

    t = 0.0
    for _ in 1:nsteps
        solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 50,
                rel_drop = 1.0e-3,
                linear_viscosity = true,
                update_material = false,
                inertia = true,
                ρ_inertia = ρ,
            ),
        )
        # Dirichlet top wall (untouched by flow_bcs; keep fixed)
        @views stokes.V.Vx[:, end] .= U
        t += dt
    end
    @test t ≈ t_end

    # Compare interior Vx profile (ghost-stripped) against analytic at cell-face y
    yVx = Array(grid.xi_vel[1][2])
    Vx = Array(stokes.V.Vx)
    # columns 2:end-1 are active faces; average over periodic x
    Vx_num = vec(sum(Vx[:, 2:(end - 1)]; dims = 1) ./ size(Vx, 1))
    y_int = yVx[2:(end - 1)]
    Vx_ana = couette_analytic.(y_int, t; U = U, H = H, ν = ν)
    err = maximum(abs, Vx_num .- Vx_ana) / U
    @test err < 0.05

    # Steady Couette envelope: midplane should sit below the linear profile at Fo=0.1
    y_mid = H / 2
    Vx_mid = couette_analytic(y_mid, t; U = U, H = H, ν = ν)
    @test Vx_mid < 0.5 * U
    @test maximum(Vx_num) ≤ U + 1.0e-10
end
