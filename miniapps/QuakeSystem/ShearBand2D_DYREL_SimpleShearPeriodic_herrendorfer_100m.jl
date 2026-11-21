# Viscosity-fold Herrendörfer on a uniform 100 m mesh.
# Domain x ∈ [-75, 75] km, y ∈ [-150, 0] km → ni = (1500, 1500).
# y = -75 km is a vertex. The fault phase is that vertex row only; a wider box
# would mark the neighbouring 100 m vertices as well.
# Wf = 100 m on the wall rock and on the fault (LaMEM fault width, not the cell size).
# Frames every 10 steps. The −50 km fault point is written every step. Stops at t = 3e9 s.
include(joinpath(@__DIR__, "ShearBand2D_DYREL_SimpleShearPeriodic_herrendorfer.jl"))

nel_x = 1500
coord_x = (-75.0e3, 75.0e3)
nel_y = 1500
coord_y = (-150.0e3, 0.0)
fault = (; x = (-75.0e3, 75.0e3), y = (-75.05e3, -74.95e3))
V_top = 4.0e-9
dt = 500.0
nsteps = 1_000_000
t_end = 3.0e9
inertia = true
ρ_inertia = nothing
nx, ny = nel_x, nel_y

RSF = (
    enabled = true,
    affect_stokes = true,
    loc = :vertex,
    V0 = 4.0e-9,
    dt_min = 1.0e-2,
    dt_max = 1.0e7,
    dt_rsf_switch = 1.0e9,
    G = 3.0e10,
    ν = 0.25,
    Phase1 = (
        a_rsf = 0.011,
        b_rsf = 0.017,
        mu0_rsf = 0.2,
        D_rs = 0.01,
        Wf = 100.0,
        state_rsf_init = 40.0,
        λ = 0.0,
        C = 0.0,
    ),
    Phase2 = (
        a_rsf = 0.011,
        b_rsf = 0.001,
        b_rsf_val = (0.001, 0.017, 0.017, 0.001),
        b_rsf_x = (-47000.0, -43000.0, 33000.0, 37000.0),
        mu0_rsf = 0.2,
        D_rs = 0.01,
        Wf = 100.0,
        state_rsf_init = -1.0,
        λ = 0.0,
        C = 0.0,
    ),
)

vtk_dir = joinpath(@__DIR__, "VTK_herrendorfer_100m_Wf100", "vtk")
VTK = (;
    do_vtk = true,
    folder = joinpath(@__DIR__, "VTK_herrendorfer_100m_Wf100"),
    vtk_dir = vtk_dir,
    name = "simple_shear_herrendorfer_100m_Wf100",
    pvd_name = "simple_shear_herrendorfer_100m_Wf100",
    vtk_every = 10,
)

println("100 m Herrendörfer: ni = ($nx, $ny), Wf = 100 m, t_end = $t_end s")
println("ParaView: $(joinpath(vtk_dir, VTK.pvd_name)).pvd   (time axis is seconds, every 10 steps)")
flush(stdout)

igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
@time main(
    igg;
    nel_x = nel_x,
    coord_x = coord_x,
    nel_y = nel_y,
    coord_y = coord_y,
    fault = fault,
    V_top = V_top,
    dt = dt,
    nsteps = nsteps,
    t_end = t_end,
    rsf_nt = RSF,
    inertia = inertia,
    ρ_inertia = ρ_inertia,
    VTK = VTK,
    probe_csv = joinpath(vtk_dir, "point_xm50km_z75000m.csv"),
    probe_xy = (-50.0e3, -75.0e3),
)
