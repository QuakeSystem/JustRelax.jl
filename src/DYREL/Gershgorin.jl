function Gershgorin_Stokes2D_SchurComplement!(
        Dx,
        Dy,
        λmaxVx,
        λmaxVy,
        η,
        ηv,
        γ_eff,
        phase_ratios,
        rheology,
        di,
        dt;
        periodic_x::Bool = false,
    )
    ni = size(η)
    @parallel (@idx ni) _Gershgorin_Stokes2D_SchurComplement!(
        Dx,
        Dy,
        λmaxVx,
        λmaxVy,
        η,
        ηv,
        γ_eff,
        di.center,
        di.vertex,
        phase_ratios.vertex,
        phase_ratios.center,
        rheology,
        dt,
        periodic_x,
    )
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di_center, di_vertex,
        phase_vertex, phase_center, rheology, dt, periodic_x
    )

    nx = size(η, 1)
    iC = i
    iCp = periodic_x ? (i == nx ? 1 : i + 1) : i + 1
    iVp = periodic_x ? (i == nx ? 1 : i + 1) : i + 1


    # @inbounds begin
    phase = phase_vertex[iVp, j + 1]
    GN = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_vertex[iVp, j]
    GS = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_center[iC, j]
    GW = fn_ratio(get_shear_modulus, rheology, phase)

    # viscosity coefficients at surrounding points
    ηN = ηv[iVp, j + 1]
    ηS = ηv[iVp, j]
    ηW = η[iC, j]
    # # bulk viscosity coefficients at surrounding points
    γW = γ_eff[iC, j]

    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)

        # Hoist common parameters
        dx = @dx(di_center, i)
        dy = @dy(di_vertex, j)
        _dx = inv(dx)
        _dy = inv(dy)
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dxdy = _dx * _dy
        c43 = 4 / 3
        c23 = 2 / 3

        phase = phase_center[iCp, j]
        GE = fn_ratio(get_shear_modulus, rheology, phase)
        ηE = η[iCp, j]
        γE = γ_eff[iCp, j]
        # effective viscoelastic viscosity
        ηN = 1 / (1 / ηN + 1 / (GN * dt))
        ηS = 1 / (1 / ηS + 1 / (GS * dt))
        ηW = 1 / (1 / ηW + 1 / (GW * dt))
        ηE = 1 / (1 / ηE + 1 / (GE * dt))

        # Precompute common terms
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        γE_dx = γE * _dx
        γW_dx = γW * _dx

        # compute Gershgorin entries
        Cxx = (ηN + ηS) * _dy2 +
            (γE + c43 * ηE) * _dx2 +
            (γW + c43 * ηW) * _dx2 +
            (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx

        Cxy = ((γE - c23 * ηE + ηN) + (γE - c23 * ηE + ηS)) * _dxdy +
            ((γW + ηN - c23 * ηW) + (γW + ηS - c23 * ηW)) * _dxdy

        # this is the preconditioner diagonal entry
        Dx_ij = Dx[i, j] = (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx
        # maximum eigenvalue estimate
        λmaxVx[i, j] = inv(Dx_ij) * (Cxx + Cxy)
    end

    # viscosity coefficients at surrounding points
    GS = GW # reuse cached value
    phase = phase_vertex[i, j + 1]
    GW = fn_ratio(get_shear_modulus, rheology, phase)
    GE = GN # reuse cached value

    # viscosity coefficients at surrounding points
    ηS = η[i, j]
    ηW = ηv[i, j + 1]
    ηE = ηv[iVp, j + 1]
    # # bulk viscosity coefficients at surrounding points
    γS = γW # reuse cached value

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        # Hoist common parameters
        dx = @dx(di_vertex, i)
        dy = @dy(di_center, j)
        _dx = inv(dx)
        _dy = inv(dy)
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dxdy = _dx * _dy
        c43 = 4 / 3
        c23 = 2 / 3

        phase = phase_center[i, j + 1]
        GN = fn_ratio(get_shear_modulus, rheology, phase)

        ηN = η[i, j + 1]
        γN = γ_eff[i, j + 1]
        # effective viscoelastic viscosity
        ηN = 1 / (1 / ηN + 1 / (GN * dt))
        ηS = 1 / (1 / ηS + 1 / (GS * dt))
        ηW = 1 / (1 / ηW + 1 / (GW * dt))
        ηE = 1 / (1 / ηE + 1 / (GE * dt))

        # Precompute common terms
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        γN_dy = γN * _dy
        γS_dy = γS * _dy

        # compute Gershgorin entries
        Cyy = (ηE + ηW) * _dx2 +
            (γN + c43 * ηN) * _dy2 +
            (γS + c43 * ηS) * _dy2 +
            (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx

        Cyx = ((γN + ηE - c23 * ηN) + (γN - c23 * ηN + ηW)) * _dxdy +
            ((γS + ηE - c23 * ηS) + (γS - c23 * ηS + ηW)) * _dxdy

        # this is the preconditioner diagonal entry
        Dy_ij = Dy[i, j] = (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
        # maximum eigenvalue estimate
        λmaxVy[i, j] = inv(Dy_ij) * (Cyx + Cyy)
    end
    # end

    return nothing
end

"""
    update_α_β!(βV, αV, dτV, cV)

Generic N-dimensional version (2D and 3D) of the acceleration parameters update.

Updates the damping parameters `βV` and `αV` for each velocity component based on
the pseudo-time step `dτV` and the preconditioner diagonal `cV`.

# Arguments
- `βV::NTuple{N, AbstractArray{T, N}}`: Tuple of damping parameters β for each velocity component
- `αV::NTuple{N, AbstractArray{T, N}}`: Tuple of acceleration parameters α for each velocity component
- `dτV::NTuple{N, AbstractArray{T, N}}`: Tuple of pseudo-time steps for each velocity component
- `cV::NTuple{N, AbstractArray{T, N}}`: Tuple of preconditioner diagonal entries for each velocity component
"""
function update_α_β!(
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}
    ni = size(βV[1]) .+ ntuple(i -> i == 1 ? 1 : 0, Val(N))
    @parallel (@idx ni) _update_α_β!(βV, αV, dτV, cV)
    return nothing
end

@parallel_indices (I...) function _update_α_β!(
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}
    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(βV[i]))
            dτV_ij = dτV[i][I...]
            cV_ij = cV[i][I...]
            βV[i][I...] = @muladd 2 * dτV_ij / (2 + cV_ij * dτV_ij)
            αV[i][I...] = @muladd (2 - cV_ij * dτV_ij) / (2 + cV_ij * dτV_ij)
        end
    end
    return nothing
end

"""
    update_dτV_α_β!(dτV, βV, αV, cV, λmaxV, CFL_v)

Generic N-dimensional version (2D and 3D) for updating pseudo-time step and acceleration parameters.

Computes the pseudo-time step `dτV` from the maximum eigenvalue estimate `λmaxV` and CFL number,
then updates the damping parameters `βV` and acceleration parameters `αV` accordingly.

# Arguments
- `dτV::NTuple{N, AbstractArray{T, N}}`: Tuple of pseudo-time steps for each velocity component
- `βV::NTuple{N, AbstractArray{T, N}}`: Tuple of damping parameters β for each velocity component
- `αV::NTuple{N, AbstractArray{T, N}}`: Tuple of acceleration parameters α for each velocity component
- `cV::NTuple{N, AbstractArray{T, N}}`: Tuple of preconditioner diagonal entries for each velocity component
- `λmaxV::NTuple{N, AbstractArray{T, N}}`: Tuple of maximum eigenvalue estimates for each velocity component
- `CFL_v::Real`: CFL number for velocity
"""
function update_dτV_α_β!(
        dτV::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}},
        λmaxV::NTuple{N, AbstractArray{T, N}},
        CFL_v::Real
    ) where {N, T}
    ni = size(βV[1]) .+ ntuple(i -> i == 1 ? 1 : 0, Val(N))
    @parallel (@idx ni) _update_dτV_α_β!(dτV, βV, αV, cV, λmaxV, CFL_v)
    return nothing
end

@parallel_indices (I...) function _update_dτV_α_β!(
        dτV::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}},
        λmaxV::NTuple{N, AbstractArray{T, N}},
        CFL_v::Real
    ) where {N, T}
    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(βV[i]))
            dτV_ij = dτV[i][I...] = 2 / √(λmaxV[i][I...]) * CFL_v
            cV_ij = cV[i][I...]
            βV[i][I...] = @muladd 2 * dτV_ij / (2 + cV_ij * dτV_ij)
            αV[i][I...] = @muladd (2 - cV_ij * dτV_ij) / (2 + cV_ij * dτV_ij)
        end
    end
    return nothing
end

# 2D wrapper for update_α_β!
function update_α_β!(dyrel::JustRelax.DYREL)
    return update_α_β!(
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.cVx, dyrel.cVy)
    )
end

# 2D wrapper for update_dτV_α_β!
function update_dτV_α_β!(dyrel::JustRelax.DYREL)
    return update_dτV_α_β!(
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.cVx, dyrel.cVy),
        (dyrel.λmaxVx, dyrel.λmaxVy),
        dyrel.CFL
    )
end

# 2D wrapper for update_α_β! with individual arguments
function update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    return update_α_β!(
        (βVx, βVy),
        (αVx, αVy),
        (dτVx, dτVy),
        (cVx, cVy)
    )
end

# 3D wrapper for update_α_β! with individual arguments
function update_α_β!(βVx, βVy, βVz, αVx, αVy, αVz, dτVx, dτVy, dτVz, cVx, cVy, cVz)
    return update_α_β!(
        (βVx, βVy, βVz),
        (αVx, αVy, αVz),
        (dτVx, dτVy, dτVz),
        (cVx, cVy, cVz)
    )
end

# 2D wrapper for update_dτV_α_β! with individual arguments
function update_dτV_α_β!(dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL_v)
    return update_dτV_α_β!(
        (dτVx, dτVy),
        (βVx, βVy),
        (αVx, αVy),
        (cVx, cVy),
        (λmaxVx, λmaxVy),
        CFL_v
    )
end

# 3D wrapper for update_dτV_α_β! with individual arguments
function update_dτV_α_β!(dτVx, dτVy, dτVz, βVx, βVy, βVz, αVx, αVy, αVz, cVx, cVy, cVz, λmaxVx, λmaxVy, λmaxVz, CFL_v)
    return update_dτV_α_β!(
        (dτVx, dτVy, dτVz),
        (βVx, βVy, βVz),
        (αVx, αVy, αVz),
        (cVx, cVy, cVz),
        (λmaxVx, λmaxVy, λmaxVz),
        CFL_v
    )
end

# # 3D wrapper for update_α_β!
# function update_α_β!(dyrel::JustRelax.DYREL)
#     return update_α_β!(
#         (dyrel.βVx,  dyrel.βVy, dyrel.βVz),
#         (dyrel.αVx,  dyrel.αVy, dyrel.αVz),
#         (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
#         (dyrel.cVx,  dyrel.cVy, dyrel.cVz)
#     )
# end

# # 3D wrapper for update_dτV_α_β!
# function update_dτV_α_β!(dyrel::JustRelax.DYREL)
#     return update_dτV_α_β!(
#         (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
#         (dyrel.βVx, dyrel.βVy, dyrel.βVz),
#         (dyrel.αVx, dyrel.αVy, dyrel.αVz),
#         (dyrel.cVx, dyrel.cVy, dyrel.cVz),
#         (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
#         dyrel.CFL
#     )
# end
