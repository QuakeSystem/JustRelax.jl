using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using Printf, LinearAlgebra, GeoParams, CellArrays
using JustRelax, JustRelax.JustRelax2D

const backend_JR = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# using JustPIC, JustPIC._2D
# # Threads is the default backend,
# # to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# # and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
# const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

@parallel_indices (i, j) function init_T!(T, z)
    if z[j] == maximum(z)
        T[i, j] = 300.0
    elseif z[j] == minimum(z)
        T[i, j] = 3500.0
    else
        T[i, j] = z[j] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        if (((x[i] - xc))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i + 1, j] += δT
        end
        return nothing
    end
    nx, ny = size(T)
    return @parallel (1:(nx - 2), 1:ny) _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end

function diffusion_2D(; nx = 64, ny = 64, lx = 100.0e3, ly = 100.0e3, ρ0 = 3.3e3, Cp0 = 1.2e3, K0 = 3.0)
    kyr = 1.0e3 * 3600 * 24 * 365.25
    Myr = 1.0e3 * kyr
    ttot = 10 * Myr # total simulation time
    dt = 500 * kyr # physical time step
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    # igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    # For non-uniform spacing we do not use `Geometry(..., di)` coordinates.
    # Instead we build x/y vertex coordinate vectors directly, then derive dx/dy
    # inverse-spacing arrays for the thermal kernels.
    origin = 0, -ly

    # Logistic-based non-uniform vertex grid generator.
    # You can control:
    # - x_center / y_center: physical center of refined zone
    # - w_ref: half-width of refined zone
    # - refine_factor: Δx_outside / Δx_center
    function logistic_vertices(
        n_cells::Int,
        L::Float64,
        x0::Float64;
        x_center::Float64,
        w_ref::Float64,
        refine_factor::Float64,
        k::Float64 = 4.0,
    )
        # Start from uniform cell widths
        widths_lin = fill(L / n_cells, n_cells)

        # Uniform-cell centers (before stretching), exactly symmetric in [x0, x0+L]
        x_centers = [x0 + (i - 0.5) * (L / n_cells) for i in 1:n_cells]

        # Logistic weight: ~1 near center (refined), → refine_factor outside
        function local_width_weight(x)
            d = abs(x - x_center) / w_ref            # 0 at center, 1 at edge of refined zone
            s = 1 / (1 + exp(-k * (d - 1.0)))        # 0 near center, →1 far outside
            return 1 + (refine_factor - 1) * s       # 1 at center, →refine_factor outside
        end

        # Apply weights based on the *uniform* centers, then renormalize once.
        widths = similar(widths_lin)
        @inbounds for i in 1:n_cells
            w = local_width_weight(x_centers[i])
            widths[i] = widths_lin[i] * w
        end

        # Renormalize so total length is exactly L
        scale = L / sum(widths)
        widths .*= scale

        vertices = zeros(n_cells + 1)
        vertices[1] = x0
        @inbounds for i in 1:n_cells
            vertices[i + 1] = vertices[i] + widths[i]
        end

        return vertices, widths
    end

    # Choose refined zone center and half-width (in physical coordinates)
    x_center = lx / 2 + 12.0e3 # shift the refined zone slightly off-center to test non-symmetric refinement
    y_center = -ly / 2 + 12.0e3
    w_ref_x = lx / 4
    w_ref_y = ly / 4
    refine_factor = 10.0

    # Build vertex coordinate vectors (used for initial conditions and perturbations)
    xvi_x, dx_cells = logistic_vertices(
        nx, lx, origin[1] * 1.0;
        x_center = x_center,
        w_ref = w_ref_x,
        refine_factor = refine_factor,
    )
    xvi_y, dy_cells = logistic_vertices(
        ny, ly, origin[2] * 1.0;
        x_center = y_center,
        w_ref = w_ref_y,
        refine_factor = refine_factor,
    )
    xvi = (xvi_x, xvi_y)

    # Cell centers (needed for plotting with physical coordinates)
    xci_x = 0.5 .* (xvi_x[1:end-1] .+ xvi_x[2:end])
    xci_y = 0.5 .* (xvi_y[1:end-1] .+ xvi_y[2:end])

    # Derive spacing arrays required by the thermal kernels.
    # Important: `DiffusionPT_solver` converts these to inverse spacing internally via `inv.(dx)` / `inv.(dy)`.
    # - For x: T has nx+3 nodes => dx_vec length nx+2 = spacing between consecutive x-nodes.
    # - For y: T has ny+1 nodes => dy_vec length ny+1 (last entry is safe padding).
    x_nodes = zeros(nx + 3)
    x_nodes[2:(nx + 2)] .= xvi_x
    x_nodes[1] = xvi_x[1] - dx_cells[1]
    x_nodes[nx + 3] = xvi_x[end] + dx_cells[end]
    dx_vec = similar(x_nodes, Float64, nx + 2)
    @inbounds for i in 1:(nx + 2)
        dx_vec[i] = x_nodes[i + 1] - x_nodes[i]
    end

    dy_vec = zeros(ny + 1)
    @inbounds for j in 1:ny
        dy_vec[j] = xvi_y[j + 1] - xvi_y[j]
    end
    dy_vec[ny + 1] = dy_vec[ny] # padding; some stencil branches may access this index

    # Stability / PT coefficient scaling: use the minimum local spacings.
    # `PTThermalCoeffs` expects (dx, dy) scalars in a tuple.
    di_eff = (minimum(dx_cells), minimum(dy_cells))

    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase = 1,
        Density = PT_Density(; ρ0 = 3.1e3, β = 0.0, T0 = 0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp0),
        Conductivity = ConstantConductivity(; k = K0),
    )
    # fields needed to compute density on the fly
    P = @zeros(ni...)
    args = (; P = P, T = @zeros(ni .+ 1...))

    ## Allocate arrays needed for every Thermal Diffusion
    thermal = ThermalArrays(backend_JR, ni)
    thermal.H .= 1.0e-6 # radiogenic heat production
    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(backend_JR, K, ρCp, dt, di_eff, li)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])


    thermal.T[:,15:20] .= 15000.0
    init_T=thermal.T
    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, -ly / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)
    temperature2center!(thermal)

    # Time loop
    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    while it < nt
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di_eff;
            kwargs = (;
                verbose = false,
                dx = dx_vec,
                dy = dy_vec,
            ),
        )

        t += dt
        it += 1
    end

    return thermal, init_T, xvi_x, xvi_y, xci_x, xci_y
end

therm, init_T, xvi_x, xvi_y, xci_x, xci_y = diffusion_2D()
using Plots
#
# Plot cell-centered temperature with physical x/y axes.
# Then overlay vertex lines to visualize the variable grid spacing.
#
plt = heatmap(
    xci_x ./ 1.0e3,
    xci_y ./ 1.0e3,
    Array(therm.Tc),
    aspect_ratio = 1,
    title = "Final Temperature (cell-centered)",
)

for xv in xvi_x
    vline!(plt, [xv ./ 1.0e3]; color = :black, alpha = 0.25, linewidth = 0.5, label = false)
end
for yv in xvi_y
    hline!(plt, [yv ./ 1.0e3]; color = :black, alpha = 0.25, linewidth = 0.5, label = false)
end

plot(plt; size = (800, 400))
