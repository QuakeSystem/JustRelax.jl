using GeophysicalModelGenerator

struct VelBox2D
    cenx::Float64
    cenz::Float64
    widthx::Float64
    widthz::Float64
    vx::Float64
    vy::Float64
    has_vx::Bool
    has_vy::Bool
end

const vel_boxes_2D = VelBox2D[]

function add_vel_box!(
    ; cenx,
      cenz,
      widthx,
      widthz,
      vx = nothing,
      vy = nothing,
)
    vx_val = vx === nothing ? 0.0 : Float64(vx)
    vy_val = vy === nothing ? 0.0 : Float64(vy)
    has_vx = vx !== nothing
    has_vy = vy !== nothing
    push!(
        vel_boxes_2D,
        VelBox2D(
            Float64(cenx),
            Float64(cenz),
            Float64(widthx),
            Float64(widthz),
            vx_val,
            vy_val,
            has_vx,
            has_vy,
        ),
    )
    return nothing
end

function GMG_subduction_2D(nx, ny)
    model_depth = 660
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz = nx, ny
    Tbot = 1474.0
    x = range(0, 3000, nx)
    air_thickness = 15.0
    z = range(-model_depth, air_thickness, nz)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx, 1, nz)
    Temp = fill(Tbot, nx, 1, nz)
    Tlab = 1300
    # lith   = LithosphericPhases(Layers=[80], Phases=[1 0], Tlab=Tlab)

    # phases
    # 0: asthenosphere
    # 1: lithosphere
    # 2: subduction lithosphere
    # 3: oceanic crust
    # 4: air
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (0, 3000),
        zlim = (-model_depth, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add left oceanic plate
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (100, 3000 - 100),
        zlim = (-model_depth, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [80], Phases = [1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add right oceanic plate crust
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (3000 - 1430, 3000 - 200),
        zlim = (-model_depth, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [8 72], Phases = [2 1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add slab
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (3000 - 1430, 3000 - 1430 - 250),
        zlim = (-80, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = -30,
        phase = LithosphericPhases(Layers = [8 80], Phases = [2 1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )
    add_vel_box!(
        cenx   = (3000 - 2430) * 1.0e3,  # m
        cenz   = -40.0 * 1.0e3,          # m
        widthx = 250.0 * 1.0e3,          # m
        widthz = 40.0 * 1.0e3,           # m
        vx     = 4.0e-9,                   # m/s (optional)
        vy     = 0.0,                    # m/s (optional)
    )
    surf = Grid2D.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 3

    Grid2D = addfield(Grid2D, (; Phases, Temp))
    write_paraview(Grid2D, "Initial_Setup_Subduction_rank")
    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :] .+ 1
    T = Temp[:, 1, :] .+ 273

    return li, origin, ph, T
end
