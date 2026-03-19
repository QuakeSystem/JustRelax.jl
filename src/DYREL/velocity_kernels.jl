## RESIDUALS

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy)
    end
    # end
    return nothing
end

# Geometry-aware overload (2D): derive local spacings from node/center vectors.
@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        P,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        x_nodes::AbstractVector,
        x_centers::AbstractVector,
        y_nodes::AbstractVector,
        y_centers::AbstractVector,
    ) where {T}
    dx_inv = inv(size_cell(x_nodes, clamp(i, 1, length(x_nodes) - 1)))
    dy_inv = inv(size_cell(y_nodes, clamp(j, 1, length(y_nodes) - 1)))
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
            d_xa_τxx = (τxx[i + 1, j] - τxx[i, j]) * dx_inv
            d_yi_τxy = (τxy[i + 1, j + 1] - τxy[i + 1, j]) * dy_inv
            d_xa_P = (P[i + 1, j] - P[i, j]) * dx_inv
            d_xa_ΔPψ = (ΔPψ[i + 1, j] - ΔPψ[i, j]) * dx_inv
            Rx[i, j] = d_xa_τxx + d_yi_τxy - d_xa_P - d_xa_ΔPψ - av_xa(ρgx)
        end
        if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
            d_ya_τyy = (τyy[i, j + 1] - τyy[i, j]) * dy_inv
            d_xi_τxy = (τxy[i + 1, j + 1] - τxy[i, j + 1]) * dx_inv
            d_ya_P = (P[i, j + 1] - P[i, j]) * dy_inv
            d_ya_ΔPψ = (ΔPψ[i, j + 1] - ΔPψ[i, j]) * dy_inv
            Ry[i, j] = d_ya_τyy + d_xi_τxy - d_ya_P - d_ya_ΔPψ - av_ya(ρgy)
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, Vx, Vy, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy, dt
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
    end

    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        θ = 1.0
        # Vertical velocity
        Vyᵢⱼ = Vy[i + 1, j + 1]
        # Get necessary buoyancy forces
        j_N = min(j + 1, ny)
        ρg_S = ρgy[i, j]
        ρg_N = ρgy[i, j_N]
        # Spatial derivatives
        ∂ρg∂y = (ρg_N - ρg_S) * _dy
        # correction term
        ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt

        Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy) + ρg_correction
    end

    return nothing
end

@parallel_indices (i, j) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, P, P_num, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, Dx, Dy, _dx, _dy
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        Rx[i, j] = (d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(P_num) - d_xa(ΔPψ) - av_xa(ρgx)) / Dx[i, j]
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        Ry[i, j] = (d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(P_num) - d_ya(ΔPψ) - av_ya(ρgy)) / Dy[i, j]
    end
    # end

    return nothing
end

# Geometry-aware overload (2D): derive local spacings from node/center vectors.
@parallel_indices (i, j) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        P,
        P_num,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        Dx,
        Dy,
        x_nodes::AbstractVector,
        x_centers::AbstractVector,
        y_nodes::AbstractVector,
        y_centers::AbstractVector,
    ) where {T}
    dx_inv = inv(size_cell(x_nodes, clamp(i, 1, length(x_nodes) - 1)))
    dy_inv = inv(size_cell(y_nodes, clamp(j, 1, length(y_nodes) - 1)))
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
            d_xa_τxx = (τxx[i + 1, j] - τxx[i, j]) * dx_inv
            d_yi_τxy = (τxy[i + 1, j + 1] - τxy[i + 1, j]) * dy_inv
            d_xa_P = (P[i + 1, j] - P[i, j]) * dx_inv
            d_xa_Pn = (P_num[i + 1, j] - P_num[i, j]) * dx_inv
            d_xa_ΔPψ = (ΔPψ[i + 1, j] - ΔPψ[i, j]) * dx_inv
            Rx[i, j] = (d_xa_τxx + d_yi_τxy - d_xa_P - d_xa_Pn - d_xa_ΔPψ - av_xa(ρgx)) / Dx[i, j]
        end
        if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
            d_ya_τyy = (τyy[i, j + 1] - τyy[i, j]) * dy_inv
            d_xi_τxy = (τxy[i + 1, j + 1] - τxy[i, j + 1]) * dx_inv
            d_ya_P = (P[i, j + 1] - P[i, j]) * dy_inv
            d_ya_Pn = (P_num[i, j + 1] - P_num[i, j]) * dy_inv
            d_ya_ΔPψ = (ΔPψ[i, j + 1] - ΔPψ[i, j]) * dy_inv
            Ry[i, j] = (d_ya_τyy + d_xi_τxy - d_ya_P - d_ya_Pn - d_ya_ΔPψ - av_ya(ρgy)) / Dy[i, j]
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, P, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, _dx, _dy, _dz
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j, k)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j, k)
    Base.@propagate_inbounds @inline av_za(A) = _av_za(A, i, j, k)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        Rx[i, j, k] = d_xa(τxx) + d_yi(τxy) + d_zi(τxz) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        Ry[i, j, k] = d_ya(τyy) + d_xi(τxy) + d_zi(τyz) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy)
    end
    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        Rz[i, j, k] = d_za(τzz) + d_xi(τxz) + d_yi(τyz) - d_za(P) - d_za(ΔPψ) - av_za(ρgz)
    end
    # end
    return nothing
end

@parallel_indices (i, j, k) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, Vx, Vy, Vz, P, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, _dx, _dy, _dz, dt
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j, k)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j, k)
    Base.@propagate_inbounds @inline av_za(A) = _av_za(A, i, j, k)

    nx, ny, nz = size(ρgz)
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        Rx[i, j, k] = d_xa(τxx) + d_yi(τxy) + d_zi(τxz) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
    end

    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        Ry[i, j, k] = d_ya(τyy) + d_xi(τxy) + d_zi(τyz) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy)
    end

    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        θ = 1.0
        # Vertical velocity
        Vzᵢⱼₖ = Vz[i + 1, j + 1, k + 1]
        # Get necessary buoyancy forces
        k_T = min(k + 1, nz)
        ρg_B = ρgz[i, j, k]
        ρg_T = ρgz[i, j, k_T]
        # Spatial derivatives
        ∂ρg∂z = (ρg_T - ρg_B) * _dz
        # correction term
        ρg_correction = (Vzᵢⱼₖ * ∂ρg∂z) * θ * dt

        Rz[i, j, k] = d_za(τzz) + d_xi(τxz) + d_yi(τyz) - d_za(P) - d_za(ΔPψ) - av_za(ρgz) + ρg_correction
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, P, P_num, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, Dx, Dy, Dz, _dx, _dy, _dz
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j, k)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j, k)
    Base.@propagate_inbounds @inline av_za(A) = _av_za(A, i, j, k)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        Rx[i, j, k] = (d_xa(τxx) + d_yi(τxy) + d_zi(τxz) - d_xa(P) - d_xa(P_num) - d_xa(ΔPψ) - av_xa(ρgx)) / Dx[i, j, k]
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        Ry[i, j, k] = (d_ya(τyy) + d_xi(τxy) + d_zi(τyz) - d_ya(P) - d_ya(P_num) - d_ya(ΔPψ) - av_ya(ρgy)) / Dy[i, j, k]
    end
    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        Rz[i, j, k] = (d_za(τzz) + d_xi(τxz) + d_yi(τyz) - d_za(P) - d_za(P_num) - d_za(ΔPψ) - av_za(ρgz)) / Dz[i, j, k]
    end
    # end

    return nothing
end

@parallel_indices (I...) function update_V_damping!(
        dVdτ::NTuple{N, AbstractArray{T, N}}, R::NTuple{N, AbstractArray{T, N}}, αV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}

    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(R[i]))
            dVdτ[i][I...] = αV[i][I...] * dVdτ[i][I...] + R[i][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function update_DR_V!(
        V::NTuple{N, AbstractArray{T, N}}, dVdτ::NTuple{N, AbstractArray{T, N}}, βV::NTuple{N, AbstractArray{T, N}}, dτV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}
    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(dVdτ[i]))
            V[i][I .+ 1...] += dVdτ[i][I...] * βV[i][I...] * dτV[i][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function update_DR_V!(
        V::NTuple{N, AbstractArray{T, N}},
        dVdτ::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
        mask_vbox::NTuple{N, AbstractArray{<:Number, N}},
    ) where {N, T}
    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(dVdτ[i]))
            # if mask==1 => prescribed region, do not update this velocity DoF
            if mask_vbox[i][I...] == 0
                V[i][I .+ 1...] += dVdτ[i][I...] * βV[i][I...] * dτV[i][I...]
            end
        end
    end

    return nothing
end
