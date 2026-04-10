using GeophysicalModelGenerator
using Statistics
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

# ------------------------------------------------------------
# Non-uniform grid helpers (logistic refinement)
# ------------------------------------------------------------

function logistic_vertices(
    n_cells::Int,
    L::Float64,
    x0::Float64;
    x_center::Float64,
    w_ref::Float64,
    refine_factor::Float64,
    k::Float64 = 4.0,
)
    # Compute cell widths with a logistic stretching weight, then integrate.
    widths_lin = fill(L / n_cells, n_cells)
    x_centers_u = [x0 + (i - 0.5) * (L / n_cells) for i in 1:n_cells]

    local_width_weight(x) = begin
        d = abs(x - x_center) / w_ref
        s = 1 / (1 + exp(-k * (d - 1.0)))
        1 + (refine_factor - 1) * s
    end

    widths = similar(widths_lin)
    @inbounds for i in 1:n_cells
        widths[i] = widths_lin[i] * local_width_weight(x_centers_u[i])
    end

    # Renormalize to preserve the total domain length exactly.
    widths .*= L / sum(widths)

    vertices = zeros(n_cells + 1)
    vertices[1] = x0
    @inbounds for i in 1:n_cells
        vertices[i + 1] = vertices[i] + widths[i]
    end

    return vertices, widths
end

function subduction_nonuniform_coords_1d(
    n_points::Int,
    x0::Float64,
    x1::Float64;
    ref_grid::Int,
    refine_factor::Float64,
    w_ref_ratio::Float64, # width of refined region as a fraction of total domain length
    x_center_frac::Float64,
    k::Float64 = 4.0, # logistic stretching parameter (higher k = sharper transition)
    verbose::Int = 0,
)
    if ref_grid == 0 || refine_factor == 1.0
    # Preserve the original type returned by `Geometry` (LinRange/StepRangeLen),
    # which downstream JustPIC advection currently dispatches on.
    return LinRange(x0, x1, n_points)
    end

    n_cells = n_points - 1
    L = x1 - x0
    x_center = x0 + x_center_frac * L
    w_ref = w_ref_ratio * abs(L)

    vertices, widths = logistic_vertices(
        n_cells,
        abs(L),
        x0;
        x_center = x_center,
        w_ref = w_ref,
        refine_factor = refine_factor,
        k = k,
    )

    if verbose == 1
        println("[grid verbose] non-uniform grid from $(x0) to $(x1):")
        println("  max cell size = $(maximum(widths))")
        println("  min cell size = $(minimum(widths))")
        println("  average cell size = $(mean(widths))")
        println("  refine_factor = $(refine_factor)")
        println("  w_ref_ratio = $(w_ref_ratio) -> w_ref = $(w_ref)")
        println("  center = $(x_center)")
    end

    return vertices
end

"""
Like `GMG_subduction_2D`, but also returns the non-uniform 1D coordinate vectors
needed by the DYREL grid setup:
`xvi` (staggered/vertex coordinates) and `xci` (cell-center coordinates) in meters.
"""
function GMG_subduction_2D_with_coords(
    nx_points::Int,
    ny_points::Int;
    ref_grid::Int = 0,
    refine_factor::Float64 = 16.0,
    w_ref_ratio::Float64 = 1 / 2, 
    k::Float64 = 16.0,
    x_center_frac::Float64 = 0.5,
    y_center_frac::Float64 = 0.5,
    verbose::Int = 1,
)
    model_depth = 50.0 # POSITIVE VALUE IN KM
    Tbot = 1474.0
    x0_km, x1_km = -25.0, 25.0
    air_thickness = 0.0
    z0_km, z1_km = -model_depth * 1.0, air_thickness

    # Our coordinate arrays are "points" for CartData: xvi has length nx_points.
    x = subduction_nonuniform_coords_1d(
        nx_points,
        x0_km,
        x1_km;
        ref_grid = ref_grid,
        refine_factor = refine_factor,
        w_ref_ratio = w_ref_ratio,
        x_center_frac = x_center_frac,
        k = k,
        verbose = verbose,
    )
    z = subduction_nonuniform_coords_1d(
        ny_points,
        z0_km,
        z1_km;
        ref_grid = ref_grid,
        refine_factor = refine_factor/4,
        w_ref_ratio = w_ref_ratio/1.5,
        x_center_frac = y_center_frac,
        k = k,
        verbose = verbose,
    )

    Grid2D = CartData(xyz_grid(x, 0, z))

    # Phases and temperature on the CartData grid points ------------------
    Phases = zeros(Int64, nx_points, 1, ny_points) .+ 4
    Temp = fill(Tbot, nx_points, 1, ny_points)
    Tlab = 1300

    # phases
    # 0: asthenosphere
    # 1: lithosphere
    # 2: subduction lithosphere
    # 3: oceanic crust
    # 4: air
    # ADD MEDIA
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (-25.0, 25.0),
        zlim = (z0_km, -25.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [1]),
    )
    # ADD PUSHING BOX
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (-10.0, 10.0),
        zlim = (-26.0, -10.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [1]),
    )

    # Add velocity strengthening zone 
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (-10.0, 10.0),
        zlim = (-25.0, -24.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [3]),
    )
        # Add velocity weaking zone 
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (-7.5, 7.5),
        zlim = (-25.0, -24.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [2]),
    )
    # Velocity box (same values as the uniform setup)
    add_vel_box!(
        cenx = 0.0,  # m
        cenz = -12.0 * 1.0e3,         # m
        widthx = 20.0 * 1.0e3,      # m
        widthz = 6.0 * 1.0e3,      # m
        vx = 4.0e-12,                  # m/s
        # vy = -4.0e-9,              # m/s (optional)
    )

    # Surface overwrite
    surf = Grid2D.z.val .> 0.0
    Temp[surf] .= 20.0
    # Phases[surf] .= 3 # air

    Grid2D = addfield(Grid2D, (; Phases, Temp))
    write_paraview(Grid2D, "Initial_Setup_Simple_Shear_rank")

    # Convert coordinates/geometry to meters for DYREL
    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    # ph = Phases[:, 1, :] .+ 1
    ph = Phases[:, 1, :]
    T = Temp[:, 1, :] .+ 273

    # Staggered grid coordinate vectors in meters:
    # - xvi are vertices (length nx_points)
    # - xci are cell centers (length nx_points-1)
    xvi = (x .* 1.0e3, z .* 1.0e3)
    xci = (
        0.5 .* (x[1:end-1] .+ x[2:end]) .* 1.0e3,
        0.5 .* (z[1:end-1] .+ z[2:end]) .* 1.0e3,
    )

    return li, origin, ph, T, xvi, xci
end
