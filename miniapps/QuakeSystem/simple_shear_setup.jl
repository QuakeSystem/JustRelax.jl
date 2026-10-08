using GeophysicalModelGenerator

# =============================================================================
# LaMEM-style segmented 1D / 2D mesh (sharp jumps between segments; optional bias)
# =============================================================================

"""
    mesh_seg_1d(nel, coord; bias=nothing) -> Vector{Float64}

Build **vertex** coordinates along one axis from LaMEM-style mesh segments
(`MeshSeg1D` / `MeshSeg1DGenCoord` in `fdstag.cpp`).

# Arguments
- `nel`: Number of **cells** per segment (`Integer` or vector of length `nseg`).
- `coord`: Segment boundary coordinates, length `nseg + 1` (including start and end).
- `bias`: Optional last/first cell-size ratio per segment (LaMEM `bias_*`). `1` (default)
  → uniform cells inside the segment (sharp size jump at interfaces). `bias ≠ 1`
  → geometric grading within that segment.

# Returns
Vertex coordinates of length `sum(nel) + 1`.
"""
function mesh_seg_1d(nel, coord; bias = nothing)
    nel_v = nel isa Integer ? (Int(nel),) : Tuple(Int(n) for n in nel)
    coord_v = Tuple(Float64(c) for c in coord)
    nseg = length(nel_v)
    length(coord_v) == nseg + 1 ||
        throw(ArgumentError("coord must have length nseg+1 = $(nseg + 1), got $(length(coord_v))"))
    any(n -> n < 1, nel_v) && throw(ArgumentError("nel entries must be ≥ 1, got $nel_v"))
    for i in 1:nseg
        coord_v[i] < coord_v[i + 1] ||
            throw(ArgumentError("coord must be strictly increasing; segment $i: $(coord_v[i]) ≥ $(coord_v[i + 1])"))
    end
    bias_v = if isnothing(bias)
        ntuple(_ -> 1.0, nseg)
    elseif bias isa Number
        ntuple(_ -> Float64(bias), nseg)
    else
        length(bias) == nseg ||
            throw(ArgumentError("bias must have length nseg = $nseg, got $(length(bias))"))
        Tuple(Float64(b) for b in bias)
    end

    nvert = sum(nel_v) + 1
    xvi = Vector{Float64}(undef, nvert)
    xvi[1] = coord_v[1]
    inode = 1
    @inbounds for iseg in 1:nseg
        M = nel_v[iseg]                 # cells in segment
        xstart = coord_v[iseg]
        xclose = coord_v[iseg + 1]
        b = bias_v[iseg]
        avgSz = (xclose - xstart) / M
        if b == 1.0
            for k in 1:M
                inode += 1
                xvi[inode] = xstart + k * avgSz
            end
        else
            # LaMEM: begSz = 2 avg/(1+bias), endSz = bias*begSz, linear size ramp
            begSz = 2 * avgSz / (1 + b)
            endSz = b * begSz
            dx = M > 1 ? (endSz - begSz) / (M - 1) : 0.0
            x = xstart
            for k in 0:(M - 1)
                x += begSz + k * dx
                inode += 1
                xvi[inode] = x
            end
        end
        xvi[inode] = xclose  # pin segment end (avoids drift)
    end
    return xvi
end

"""
    segmented_grid_2D(; nel_x, coord_x, nel_y, coord_y, bias_x=nothing, bias_y=nothing)

LaMEM 2D mesh from per-axis segments (`nel_*`, `coord_*`). In LaMEM the refined
axis is often `z`; here that is **`y`**.

# Keywords
- `nel_x`, `nel_y`: cells per segment (scalar or vector).
- `coord_x`, `coord_y`: segment boundaries (length = nseg + 1).
- `bias_x`, `bias_y`: optional per-segment bias (default uniform within segment).

# Returns
NamedTuple `(; xvi, ni, li, origin, di_vertex)` where `xvi = (xv, yv)`,
`ni = (nx, ny)` cell counts, and `di_vertex` are per-cell vertex spacings.
"""
function segmented_grid_2D(;
        nel_x,
        coord_x,
        nel_y,
        coord_y,
        bias_x = nothing,
        bias_y = nothing,
    )
    xv = mesh_seg_1d(nel_x, coord_x; bias = bias_x)
    yv = mesh_seg_1d(nel_y, coord_y; bias = bias_y)
    ni = (length(xv) - 1, length(yv) - 1)
    origin = (xv[1], yv[1])
    li = (xv[end] - xv[1], yv[end] - yv[1])
    di_vertex = (diff(xv), diff(yv))
    return (; xvi = (xv, yv), ni, li, origin, di_vertex)
end

# =============================================================================
# Phase setup on an existing staggered mesh
# =============================================================================

"""
    simple_shear_2D(xvi_or_npts...; ...)

Build a 2D box with a horizontal fault band via GMG `add_box!`.

# Forms
- `simple_shear_2D(nx_pts, ny_pts, coord_x, coord_y, Temp, fault, VTK)` —
  uniform vertex grid (`nx_pts`/`ny_pts` are **vertex** counts).
- `simple_shear_2D(xvi::NTuple{2,AbstractVector}, Temp, fault, VTK)` —
  use explicit vertex coordinates (e.g. from [`segmented_grid_2D`](@ref)).
"""
function simple_shear_2D(nx_pts::Integer, ny_pts::Integer, coord_x, coord_y, Temp_constant, fault, VTK)
    x = range(Float64(coord_x[1]), Float64(coord_x[2]), Int(nx_pts))
    z = range(Float64(coord_y[1]), Float64(coord_y[2]), Int(ny_pts))
    return simple_shear_2D((collect(x), collect(z)), Temp_constant, fault, VTK)
end

function simple_shear_2D(xvi::NTuple{2, <:AbstractVector}, Temp_constant, fault, VTK)
    x = collect(Float64, xvi[1])
    z = collect(Float64, xvi[2])
    nx_pts, ny_pts = length(x), length(z)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx_pts, 1, ny_pts)
    Temp = fill(Float64(Temp_constant), nx_pts, 1, ny_pts)
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
    xvi_out = (x, z)
    xci = (
        0.5 .* (xvi_out[1][1:(end - 1)] .+ xvi_out[1][2:end]),
        0.5 .* (xvi_out[2][1:(end - 1)] .+ xvi_out[2][2:end]),
    )
    staggered_grid = (; li, xvi = xvi_out, xci, origin = (first(x), first(z)))
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

Phase maps are built on the host, then copied to the ParallelStencil backend
(`Data.Array`) so the fill kernel is valid on CUDA.
"""
function init_phase_ratios_from_grid!(phase_ratios, ph_vertex, nphases::Integer)
    nx, ny = size(phase_ratios.center)
    ph_c_h = Matrix{Int}(undef, nx, ny)
    ph_v_h = Matrix{Int}(undef, nx + 1, ny + 1)
    nvx, nvy = size(ph_vertex)
    @inbounds for j in 1:ny, i in 1:nx
        ph_c_h[i, j] = Int(ph_vertex[min(i, nvx), min(j, nvy)])
    end
    @inbounds for j in 1:(ny + 1), i in 1:(nx + 1)
        ph_v_h[i, j] = Int(ph_vertex[min(i, nvx), min(j, nvy)])
    end
    ph_c = Data.Array(ph_c_h)
    ph_v = Data.Array(ph_v_h)
    @parallel (@idx (nx, ny)) _set_phases_from_map!(phase_ratios.center, ph_c, nphases)
    @parallel (@idx (nx + 1, ny + 1)) _set_phases_from_map!(phase_ratios.vertex, ph_v, nphases)
    return nothing
end
