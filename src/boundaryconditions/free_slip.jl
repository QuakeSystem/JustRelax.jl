@parallel_indices (i) function free_slip!(Ax, Ay, bc)
    @inbounds begin
        if i ≤ size(Ax, 1)
            bc.bot && (Ax[i, 1] = Ax[i, 2])
            bc.top && (Ax[i, end] = Ax[i, end - 1])
        end
        if i ≤ size(Ay, 2)
            bc.left && (Ay[1, i] = Ay[2, i])
            bc.right && (Ay[end, i] = Ay[end - 1, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function free_slip!(Ax, Ay, Az, bc)
    @inbounds begin
        # free slip in the front and back XZ planes
        if bc.front
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, 1, j] = Ax[i, 2, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, 1, j] = Az[i, 2, j]
            end
        end
        if bc.back
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, end, j] = Ax[i, end - 1, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, end, j] = Az[i, end - 1, j]
            end
        end
        # free slip in the top and bottom XY planes
        if bc.top
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, end] = Ax[i, j, end - 1]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, end] = Ay[i, j, end - 1]
            end
        end
        if bc.bot
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, 1] = Ax[i, j, 2]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, 1] = Ay[i, j, 2]
            end
        end
        # free slip in the left and right YZ planes
        if bc.left
            if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
                Ay[1, i, j] = Ay[2, i, j]
            end
            if i ≤ size(Az, 2) && j ≤ size(Az, 3)
                Az[1, i, j] = Az[2, i, j]
            end
        end
        if bc.right
            if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
                Ay[end, i, j] = Ay[end - 1, i, j]
            end
            if i ≤ size(Az, 2) && j ≤ size(Az, 3)
                Az[end, i, j] = Az[end - 1, i, j]
            end
        end
    end
    return nothing
end

@parallel_indices (i) function free_slip!(T::_T, bc) where {_T <: AbstractArray{<:Any, 2}}
    @inbounds begin
        if i ≤ size(T, 1)
            bc.bot && (T[i, 1] = T[i, 2])
            bc.top && (T[i, end] = T[i, end - 1])
        end
        if i ≤ size(T, 2)
            bc.left && (T[1, i] = T[2, i])
            bc.right && (T[end, i] = T[end - 1, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function free_slip!(T::_T, bc) where {_T <: AbstractArray{<:Any, 3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            bc.bot && (T[i, j, 1] = T[i, j, 2])
            bc.top && (T[i, j, end] = T[i, j, end - 1])
        end
        if i ≤ ny && j ≤ nz
            bc.left && (T[1, i, j] = T[2, i, j])
            bc.right && (T[end, i, j] = T[end - 1, i, j])
        end
        if i ≤ nx && j ≤ nz
            bc.front && (T[i, 1, j] = T[i, 2, j])
            bc.back && (T[i, end, j] = T[i, end - 1, j])
        end
    end
    return nothing
end

# Strike-slip faces are boundary nodes the momentum kernel does not write. A free
# surface there is a Neumann condition on the strike velocity, ∂Vx/∂x = 0, not the
# in-plane free-slip copy of the tangential component.
@parallel_indices (j) function _neumann_vx_right!(Vx)
    @inbounds Vx[end, j] = Vx[end - 1, j]
    return nothing
end

@parallel_indices (j) function _neumann_vx_left!(Vx)
    @inbounds Vx[1, j] = Vx[2, j]
    return nothing
end

# Ghost value whose average with the first interior node is the wall velocity.
@parallel_indices (i) function _vx_wall_top!(Vx, Vwall)
    @inbounds Vx[i, end] = 2 * Vwall - Vx[i, end - 1]
    return nothing
end

@parallel_indices (i) function _vx_wall_bot!(Vx, Vwall)
    @inbounds Vx[i, 1] = 2 * Vwall - Vx[i, 2]
    return nothing
end

"""
    apply_boundary_vx!(Vx, bc)

Apply strike-slip boundary conditions that [`VelocityBoundaryConditions`](@ref)
does not express. `bc = nothing` does nothing. Otherwise `bc` is a named tuple:

- `free_surface = :right` or `:left` copies the adjacent interior `Vx` onto that
  face (`∂Vx/∂x = 0`).
- `top` and `bot`, when set to a number, are wall velocities. The ghost is
  `2 Vwall − V_interior`.

The free-surface copy runs first and the wall ghosts second, so a corner follows
the wall.
"""
function apply_boundary_vx!(Vx, bc)
    bc === nothing && return nothing
    ndims(Vx) == 2 || error("apply_boundary_vx! is implemented for 2D")
    side = get(bc, :free_surface, :none)
    if side === :right
        @parallel (@idx size(Vx, 2)) _neumann_vx_right!(Vx)
    elseif side === :left
        @parallel (@idx size(Vx, 2)) _neumann_vx_left!(Vx)
    elseif side !== :none
        error("boundary_vx.free_surface must be :none, :left, or :right")
    end
    top = get(bc, :top, nothing)
    bot = get(bc, :bot, nothing)
    top === nothing ||
        @parallel (@idx size(Vx, 1)) _vx_wall_top!(Vx, convert(eltype(Vx), top))
    bot === nothing ||
        @parallel (@idx size(Vx, 1)) _vx_wall_bot!(Vx, convert(eltype(Vx), bot))
    return nothing
end
