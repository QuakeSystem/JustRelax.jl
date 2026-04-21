@parallel_indices (i) function periodic_x!(Ax, Ay)
    @inbounds begin
        # x-normal component: treat seam columns as periodic ghosts from interior.
        # This avoids pinning Ax[1,:] to a wall-like value (often zero) and then
        # mirroring it to Ax[end,:].
        if i <= size(Ax, 2)
            Ax[1, i] = Ax[end - 1, i]
            Ax[end, i] = Ax[2, i]
        end
        # y-normal component (ghosted in x): wrap x-ghosts
        if i <= size(Ay, 2)
            Ay[1, i] = Ay[end - 1, i]
            Ay[end, i] = Ay[2, i]
        end
    end
    return nothing
end

@parallel_indices (i) function periodic_x!(T::_T) where {_T <: AbstractArray{<:Any, 2}}
    @inbounds begin
        if i <= size(T, 2)
            T[1, i] = T[end - 1, i]
            T[end, i] = T[2, i]
        end
    end
    return nothing
end

