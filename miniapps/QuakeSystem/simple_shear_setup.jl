using GeophysicalModelGenerator

"""
    simple_shear_2D(nx_pts, ny_pts, coord_x, coord_y, Temp_constant, fault, VTK)

Build a 2D box with a horizontal fault band via GMG `add_box!`.

`nx_pts`/`ny_pts` are **vertex** counts. Returns staggered coordinate vectors,
integer phase map on vertices (`1` = media, `2` = fault), and temperature.
"""
function simple_shear_2D(nx_pts, ny_pts, coord_x, coord_y, Temp_constant, fault, VTK)
    x = range(coord_x[1], coord_x[2], nx_pts)
    z = range(coord_y[1], coord_y[2], ny_pts)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx_pts, 1, ny_pts)
    Temp = fill(Temp_constant, nx_pts, 1, ny_pts)
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
    if get(VTK, :do_vtk, false) && hasproperty(VTK, :name)
        write_paraview(Grid2D, VTK.name; directory = VTK.folder)
    end
    li = (abs(last(x) - first(x)), abs(last(z) - first(z)))
    xvi = (collect(x), collect(z))
    xci = (
        0.5 .* (xvi[1][1:(end - 1)] .+ xvi[1][2:end]),
        0.5 .* (xvi[2][1:(end - 1)] .+ xvi[2][2:end]),
    )
    staggered_grid = (; li, xvi, xci, origin = (first(x), first(z)))
    # GMG phases: 0 = media, 1 = fault → JustRelax phases 1, 2
    ph = Phases[:, 1, :] .+ 1
    T = Temp[:, 1, :]
    return staggered_grid, ph, T
end

@parallel_indices (i, j) function _set_phases_from_map!(phases, phmap, nph)
    p = phmap[i, j]
    for ip in 1:nph
        @index phases[ip, i, j] = ifelse(ip == p, 1.0, 0.0)
    end
    return nothing
end

"""
Fill Eulerian `PhaseRatios` from a vertex phase map (`ph[ivx, ivy]`).
"""
function init_phase_ratios_from_grid!(phase_ratios, ph_vertex, nphases::Integer)
    nx, ny = size(phase_ratios.center)
    ph_c = Matrix{Int}(undef, nx, ny)
    ph_v = Matrix{Int}(undef, nx + 1, ny + 1)
    nvx, nvy = size(ph_vertex)
    @inbounds for j in 1:ny, i in 1:nx
        ph_c[i, j] = Int(ph_vertex[min(i, nvx), min(j, nvy)])
    end
    @inbounds for j in 1:(ny + 1), i in 1:(nx + 1)
        ph_v[i, j] = Int(ph_vertex[min(i, nvx), min(j, nvy)])
    end
    @parallel (@idx (nx, ny)) _set_phases_from_map!(phase_ratios.center, ph_c, nphases)
    @parallel (@idx (nx + 1, ny + 1)) _set_phases_from_map!(phase_ratios.vertex, ph_v, nphases)
    return nothing
end
