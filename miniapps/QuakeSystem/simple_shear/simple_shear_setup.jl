using GeophysicalModelGenerator

function simple_shear_2D(nx, ny, coord_x, coord_y, Temp_constant, fault, VTK)
 # Create simple box with fault zone in the middle - fault zone is described by 1 phase == 1
    nx, nz = nx, ny
    x = range(coord_x[1], coord_x[2], nx)
    z = range(coord_y[1], coord_y[2], ny)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx, 1, nz)
    Temp = fill(Temp_constant, nx, 1, nz)
    # Add fault zone
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (fault.x[1], fault.x[2]),
        zlim = (fault.y[1], fault.y[2]),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [1], Tlab = Temp_constant),
    )
    Grid2D = addfield(Grid2D, (; Phases, Temp)) 
    if VTK.do_vtk
        write_paraview(Grid2D, VTK.name; directory = VTK.folder)
    end
    li = (abs(last(x) - first(x)), abs(last(z) - first(z)))
    # Staggered grid coordinate vectors in meters:
    # - xvi are vertices (length nx_points)
    # - xci are cell centers (length nx_points-1)
    xvi = (x, z)
    xci = (
        0.5 .* (x[1:end-1] .+ x[2:end]),
        0.5 .* (z[1:end-1] .+ z[2:end]),
    )
    staggered_grid = (; li,xvi, xci)
    ph = Phases[:, 1, :] .+ 1 
    T = Temp[:, 1, :]

    return staggered_grid, ph, T
    
end