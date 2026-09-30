@parallel_indices (i) function periodic_boundary!(T::_T, bc) where {_T <: AbstractArray{<:Any, 2}}
    @inbounds begin
        if i ≤ size(T, 1)
            bc.bot && (T[i, 1] = T[i, end - 1])
            bc.top && (T[i, end] = T[i, 2])
        end
        if i ≤ size(T, 2)
            bc.left && (T[1, i] = T[end - 1, i])
            bc.right && (T[end, i] = T[2, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function periodic_boundary!(T::_T, bc) where {_T <: AbstractArray{<:Any, 3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            bc.bot && (T[i, j, 1] = T[i, j, end - 1])
            bc.top && (T[i, j, end] = T[i, j, 2])
        end
        if i ≤ ny && j ≤ nz
            bc.left && (T[1, i, j] = T[end - 1, i, j])
            bc.right && (T[end, i, j] = T[2, i, j])
        end
        if i ≤ nx && j ≤ nz
            bc.front && (T[i, 1, j] = T[i, end - 1, j])
            bc.back && (T[i, end, j] = T[i, 2, j])
        end
    end
    return nothing
end

# Staggered velocity: x-periodicity wraps both Vx (vertical faces) and Vy (x-ghosted).
# Indexing matches the proven JR_dev simple-shear kernel:
#   Vx[1,:] = Vx[end-1,:], Vx[end,:] = Vx[2,:]
#   Vy[1,:] = Vy[end-1,:], Vy[end,:] = Vy[2,:]
@parallel_indices (i) function periodic_boundary!(Ax, Ay, bc)
    @inbounds begin
        if bc.left && bc.right
            if i ≤ size(Ax, 2)
                Ax[1, i] = Ax[end - 1, i]
                Ax[end, i] = Ax[2, i]
            end
            if i ≤ size(Ay, 2)
                Ay[1, i] = Ay[end - 1, i]
                Ay[end, i] = Ay[2, i]
            end
        end
        if bc.bot && bc.top
            if i ≤ size(Ax, 1)
                Ax[i, 1] = Ax[i, end - 1]
                Ax[i, end] = Ax[i, 2]
            end
            if i ≤ size(Ay, 1)
                Ay[i, 1] = Ay[i, end - 1]
                Ay[i, end] = Ay[i, 2]
            end
        end
    end
    return nothing
end

"""
    apply_prescribed_velocity!(Vx, Vy, prescribed)

Apply optional wall velocity prescriptions after face BCs.
Currently supports `prescribed.top_Vx` (simple-shear top drive).
"""
@views function apply_prescribed_velocity!(Vx, Vy, prescribed)
    if hasproperty(prescribed, :top_Vx)
        Vtop = getproperty(prescribed, :top_Vx)
        if !isnothing(Vtop)
            # Dirichlet ghost for wall value Vtop (same as JR_dev apply_top_shear_bc!)
            Vx[:, end] .= 2 .* Vtop .- Vx[:, end - 1]
            Vy[:, end] .= zero(eltype(Vy))
        end
    end
    return nothing
end
