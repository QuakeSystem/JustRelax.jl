# Uniform 250 m Herrendörfer eigenstrain. Same fault and loading as the graded run.
# Every cell is 250 m. Wf stays the 500 m slip-law width.
include(joinpath(@__DIR__, "ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer_eigenstrain.jl"))

mesh = herrendorfer_uniform_mesh(; h = 250.0)
println("uniform 250 m   ni = $(mesh.ni)")
flush(stdout)
nx, ny = mesh.ni
igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
main(
    igg;
    xvi = mesh.xvi,
    Wf = 500.0,
    y_fault = -75.0e3,
    nsteps = 1_000_000,
    n_ruptures = 5,
    max_years = 600.0,
    vtk_dir = joinpath(@__DIR__, "VTK_eigenstrain_herrendorfer_uniform250", "vtk"),
    pvd_name = "herrendorfer_eigenstrain_uniform250",
)
