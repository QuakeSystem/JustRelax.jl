# Load script dependencies
using GeoParams, GLMakie

const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg; Pkg.activate("miniapps")

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load file with all the rheology configurations
include("helper_functions.jl")
include("simple_shear_setup.jl")
include("simple_shear_rheology.jl")
include("main.jl")

# OUTPUT
vtk_name = "simple_shear"
vtk_folder = joinpath(@__DIR__, "VTK")
VTK = (;
    name = vtk_name,
    folder = vtk_folder,
    path = joinpath(vtk_folder, vtk_name),
    do_vtk = true,                 # ParaView VTK + PVD
    pictures = false,               # Makie PNG snapshots
    fig_dir = joinpath(vtk_folder, "figs"),
    vtk_dir = joinpath(vtk_folder, "vtk"),
    checkpoint_dir = joinpath(vtk_folder, "checkpoint"),
    pvd_name = vtk_name,
    vtk_every = 1,
    picture_every = 1,
    save_particle_points = false,  # large particle point-cloud VTKs
    particle_vtk_every = 50,
    quiet_runtime = false,
)
periodic = true
shear_rate_top = 4.0e-9  # m/s top-wall Vx
dt = 500.0               # fixed physical timestep [s]
# size of the domain
nx   =  300
ny   =  300
# Extents of the domain in meters
coord_x =  -75000, 75000
coord_y =  -150000, 0
fault_x = -75000, 75000
fault_y = -75500, -74500
fault = (; x = fault_x, y = fault_y)
Temp_constant = 1250.0
staggered_grid, phases_GMG, T_GMG = simple_shear_2D(nx + 1, ny + 1, coord_x, coord_y, Temp_constant, fault, VTK)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(
    staggered_grid, phases_GMG, T_GMG, igg;
    nx = nx, ny = ny, periodic = periodic, shear_rate_top = shear_rate_top, dt = dt,
);
