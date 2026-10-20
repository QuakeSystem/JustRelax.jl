function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
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
    )
    return nothing
end

"""
    apply_free_surface_diagonal!(Dn, λmaxVn, ρgn, di_center, dt)

Add the free-surface diagonal `-dt * ∂n(ρg)` to the normal DYREL preconditioner
and its Gershgorin row bound. The normal direction is y in 2D and z in 3D.
Passing `dt = 0` is a no-op.
"""
function apply_free_surface_diagonal!(Dy::AbstractArray{<:Any, 2}, λmaxVy, ρgy, di_center, dt)
    ni = size(Dy)
    @parallel (@idx ni) _apply_free_surface_diagonal_2D!(Dy, λmaxVy, ρgy, di_center, dt)
    return nothing
end

@parallel_indices (i, j) function _apply_free_surface_diagonal_2D!(Dy, λmaxVy, ρgy, di_center, dt)
    @inbounds if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        _dy = inv(@dy(di_center, j))
        j_N = min(j + 1, size(ρgy, 2))
        c_fs = free_surface_diagonal(ρgy[i, j], ρgy[i, j_N], _dy, dt)
        D_old = Dy[i, j]
        row_sum = λmaxVy[i, j] * D_old + c_fs
        D_new = D_old + c_fs
        Dy[i, j] = D_new
        λmaxVy[i, j] = row_sum / D_new
    end
    return nothing
end

function apply_free_surface_diagonal!(Dz::AbstractArray{<:Any, 3}, λmaxVz, ρgz, di_center, dt)
    ni = size(Dz)
    @parallel (@idx ni) _apply_free_surface_diagonal_3D!(Dz, λmaxVz, ρgz, di_center, dt)
    return nothing
end

@parallel_indices (i, j, k) function _apply_free_surface_diagonal_3D!(Dz, λmaxVz, ρgz, di_center, dt)
    @inbounds if i ≤ size(Dz, 1) && j ≤ size(Dz, 2) && k ≤ size(Dz, 3)
        _dz = inv(@dz(di_center, k))
        k_T = min(k + 1, size(ρgz, 3))
        c_fs = free_surface_diagonal(ρgz[i, j, k], ρgz[i, j, k_T], _dz, dt)
        D_old = Dz[i, j, k]
        row_sum = λmaxVz[i, j, k] * D_old + c_fs
        D_new = D_old + c_fs
        Dz[i, j, k] = D_new
        λmaxVz[i, j, k] = row_sum / D_new
    end
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di_center, di_vertex,
        phase_vertex, phase_center, rheology, dt
    )


    # @inbounds begin
    phase = phase_vertex[i + 1, j + 1]
    GN = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_vertex[i + 1, j]
    GS = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_center[i, j]
    GW = fn_ratio(get_shear_modulus, rheology, phase)

    # viscosity coefficients at surrounding points
    ηN = ηv[i + 1, j + 1]
    ηS = ηv[i + 1, j]
    ηW = η[i, j]
    # # bulk viscosity coefficients at surrounding points
    γW = γ_eff[i, j]

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

        # `Dx` reaches `i = size(η, 1)` only when the x-direction is periodic; that row belongs to
        # the seam face, whose eastern cell is the first one. The vertex reads above are already
        # right there -- `ηv[size(η, 1) + 1, :]` is the seam vertex.
        iE = wrap_next(i, size(η, 1))
        phase = phase_center[iE, j]
        GE = fn_ratio(get_shear_modulus, rheology, phase)
        ηE = η[iE, j]
        γE = γ_eff[iE, j]
        # effective viscoelastic viscosity
        # Equivalent to `inv(inv(η) + inv(G * dt))`, while preserving the
        # `G == Inf` limit (`ηve == η`) without producing `Inf / Inf`.
        ηN = ηN / @muladd(1 + ηN * inv(GN * dt))
        ηS = ηS / @muladd(1 + ηS * inv(GS * dt))
        ηW = ηW / @muladd(1 + ηW * inv(GW * dt))
        ηE = ηE / @muladd(1 + ηE * inv(GE * dt))

        # Precompute common terms
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        γE_dx = γE * _dx
        γW_dx = γW * _dx

        # compute Gershgorin entries
        Cxx = @muladd abs(ηN * _dy2) +
            abs(ηS * _dy2) +
            abs((γE + c43 * ηE) * _dx2) +
            abs((γW + c43 * ηW) * _dx2) +
            abs((ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx)

        Cxy = @muladd abs((γE - c23 * ηE + ηN) * _dxdy) +
            abs((γE - c23 * ηE + ηS) * _dxdy) +
            abs((γW + ηN - c23 * ηW) * _dxdy) +
            abs((γW + ηS - c23 * ηW) * _dxdy)

        # this is the preconditioner diagonal entry
        Dx_ij = Dx[i, j] = @muladd (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx
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
    ηE = ηv[i + 1, j + 1]
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

        # Counterpart of `iE` in the x-block: the northern cell of the y seam face is the first.
        jN = wrap_next(j, size(η, 2))
        phase = phase_center[i, jN]
        GN = fn_ratio(get_shear_modulus, rheology, phase)

        ηN = η[i, jN]
        γN = γ_eff[i, jN]
        # effective viscoelastic viscosity
        ηN = ηN / @muladd(1 + ηN * inv(GN * dt))
        ηS = ηS / @muladd(1 + ηS * inv(GS * dt))
        ηW = ηW / @muladd(1 + ηW * inv(GW * dt))
        ηE = ηE / @muladd(1 + ηE * inv(GE * dt))

        # Precompute common terms
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        γN_dy = γN * _dy
        γS_dy = γS * _dy

        # compute Gershgorin entries
        Cyy = @muladd abs(ηE * _dx2) +
            abs(ηW * _dx2) +
            abs((γN + c43 * ηN) * _dy2) +
            abs((γS + c43 * ηS) * _dy2) +
            abs((γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx)

        Cyx = @muladd abs((γN + ηE - c23 * ηN) * _dxdy) +
            abs((γN - c23 * ηN + ηW) * _dxdy) +
            abs((γS + ηE - c23 * ηS) * _dxdy) +
            abs((γS - c23 * ηS + ηW) * _dxdy)

        # this is the preconditioner diagonal entry
        Dy_ij = Dy[i, j] = @muladd (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
        # maximum eigenvalue estimate
        λmaxVy[i, j] = inv(Dy_ij) * (Cyx + Cyy)
    end
    # end

    return nothing
end

# Tangent viscosity already includes elasticity and creep. A non-positive or
# non-finite sample falls back to the Maxwell combination of the creep buffer.
@inline function _η_gershgorin(ηij, G, dt, ηtan, mask)
    if mask > 0.5 && isfinite(ηtan) && ηtan > 0
        return ηtan
    end
    return ηij / @muladd(1 + ηij * inv(G * dt))
end

function Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt,
        η_tan, η_tan_v, rsf_mask, rsf_mask_v,
    )
    ni = size(η)
    @parallel (@idx ni) _Gershgorin_Stokes2D_RSF!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di.center, di.vertex,
        phase_ratios.vertex, phase_ratios.center, rheology, dt,
        η_tan, η_tan_v, rsf_mask, rsf_mask_v,
    )
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_RSF!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di_center, di_vertex,
        phase_vertex, phase_center, rheology, dt,
        η_tan, η_tan_v, rsf_mask, rsf_mask_v,
    )
    phase = phase_vertex[i + 1, j + 1]
    GN = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_vertex[i + 1, j]
    GS = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_center[i, j]
    GW = fn_ratio(get_shear_modulus, rheology, phase)

    ηN = _η_gershgorin(ηv[i + 1, j + 1], GN, dt, η_tan_v[i + 1, j + 1], rsf_mask_v[i + 1, j + 1])
    ηS = _η_gershgorin(ηv[i + 1, j], GS, dt, η_tan_v[i + 1, j], rsf_mask_v[i + 1, j])
    ηW = _η_gershgorin(η[i, j], GW, dt, η_tan[i, j], rsf_mask[i, j])
    γW = γ_eff[i, j]

    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
        dx = @dx(di_center, i)
        dy = @dy(di_vertex, j)
        _dx = inv(dx)
        _dy = inv(dy)
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dxdy = _dx * _dy
        c43 = 4 / 3
        c23 = 2 / 3

        iE = wrap_next(i, size(η, 1))
        phase = phase_center[iE, j]
        GE = fn_ratio(get_shear_modulus, rheology, phase)
        ηE = _η_gershgorin(η[iE, j], GE, dt, η_tan[iE, j], rsf_mask[iE, j])
        γE = γ_eff[iE, j]

        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        γE_dx = γE * _dx
        γW_dx = γW * _dx

        Cxx = @muladd abs(ηN * _dy2) +
            abs(ηS * _dy2) +
            abs((γE + c43 * ηE) * _dx2) +
            abs((γW + c43 * ηW) * _dx2) +
            abs((ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx)

        Cxy = @muladd abs((γE - c23 * ηE + ηN) * _dxdy) +
            abs((γE - c23 * ηE + ηS) * _dxdy) +
            abs((γW + ηN - c23 * ηW) * _dxdy) +
            abs((γW + ηS - c23 * ηW) * _dxdy)

        Dx_ij = Dx[i, j] = @muladd (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx
        λmaxVx[i, j] = inv(Dx_ij) * (Cxx + Cxy)
    end

    GS = GW
    phase = phase_vertex[i, j + 1]
    GW = fn_ratio(get_shear_modulus, rheology, phase)
    GE = GN

    ηS = _η_gershgorin(η[i, j], GS, dt, η_tan[i, j], rsf_mask[i, j])
    ηW = _η_gershgorin(ηv[i, j + 1], GW, dt, η_tan_v[i, j + 1], rsf_mask_v[i, j + 1])
    ηE = _η_gershgorin(ηv[i + 1, j + 1], GE, dt, η_tan_v[i + 1, j + 1], rsf_mask_v[i + 1, j + 1])
    γS = γW

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        dx = @dx(di_vertex, i)
        dy = @dy(di_center, j)
        _dx = inv(dx)
        _dy = inv(dy)
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dxdy = _dx * _dy
        c43 = 4 / 3
        c23 = 2 / 3

        jN = wrap_next(j, size(η, 2))
        phase = phase_center[i, jN]
        GN = fn_ratio(get_shear_modulus, rheology, phase)
        ηN = _η_gershgorin(η[i, jN], GN, dt, η_tan[i, jN], rsf_mask[i, jN])
        γN = γ_eff[i, jN]

        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        γN_dy = γN * _dy
        γS_dy = γS * _dy

        Cyy = @muladd abs(ηE * _dx2) +
            abs(ηW * _dx2) +
            abs((γN + c43 * ηN) * _dy2) +
            abs((γS + c43 * ηS) * _dy2) +
            abs((γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx)

        Cyx = @muladd abs((γN + ηE - c23 * ηN) * _dxdy) +
            abs((γN - c23 * ηN + ηW) * _dxdy) +
            abs((γS + ηE - c23 * ηS) * _dxdy) +
            abs((γS - c23 * ηS + ηW) * _dxdy)

        Dy_ij = Dy[i, j] = @muladd (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
        λmaxVy[i, j] = inv(Dy_ij) * (Cyx + Cyy)
    end
    return nothing
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{2}, Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt,
        η_tan, η_tan_v, rsf_mask, rsf_mask_v,
    )
    return Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt,
        η_tan, η_tan_v, rsf_mask, rsf_mask_v,
    )
end

# Drop factor η_prev/η_tan on masked vertices (> 1 when the tangent softened).
# Unmasked vertices and a zero previous sample contribute 1.
@parallel_indices (i, j) function _tangent_drop_ratio!(drop, η_tan_v, η_prev_v, rsf_mask_v)
    r = one(eltype(drop))
    if rsf_mask_v[i, j] > 0.5
        ηn = η_tan_v[i, j]
        ηo = η_prev_v[i, j]
        if isfinite(ηn) && isfinite(ηo) && ηn > 0 && ηo > 0
            r = ηo / ηn
        end
    end
    drop[i, j] = r
    return nothing
end

function max_tangent_drop!(drop, η_tan_v, η_prev_v, rsf_mask_v)
    @parallel (@idx size(η_tan_v)) _tangent_drop_ratio!(drop, η_tan_v, η_prev_v, rsf_mask_v)
    return maximum_mpi(drop)
end

@parallel_indices (i, j) function _masked_positive!(out, η, mask, empty)
    ηij = η[i, j]
    out[i, j] = (mask[i, j] > 0.5 && isfinite(ηij) && ηij > 0) ? ηij : empty
    return nothing
end

# Where the penalty dominates the tangent, λmax is a ratio and does not shrink
# dτ after the fault softens. Scale that node's pseudo-step by √(η_new/η_old), but
# never below `scale_min` per rebuild: the comparison is against the previous rebuild,
# so a many-decade drop would otherwise freeze the node (dτ ~ 1e-27) for the rest of the
# pass, and a frozen velocity cannot relieve the stress that caused the drop.
@parallel_indices (i, j) function _shrink_dτ_rsf!(
        dτVx, dτVy, αVx, αVy, βVx, βVy, cVx, cVy,
        γ_eff, η_tan_v, η_prev_v, rsf_mask_v, γ_ratio, scale_min,
    )
    if i ≤ size(dτVx, 1) && j ≤ size(dτVx, 2)
        scale = one(eltype(dτVx))
        iv = i + 1
        if iv ≤ size(rsf_mask_v, 1)
            jS = j
            jN = j + 1
            γW = γ_eff[min(i, size(γ_eff, 1)), min(j, size(γ_eff, 2))]
            iE = min(i + 1, size(γ_eff, 1))
            γE = γ_eff[iE, min(j, size(γ_eff, 2))]
            γrow = max(γW, γE)
            if jS ≤ size(rsf_mask_v, 2)
                scale = _soften_scale(scale, rsf_mask_v[iv, jS], η_tan_v[iv, jS], η_prev_v[iv, jS], γrow, γ_ratio, scale_min)
            end
            if jN ≤ size(rsf_mask_v, 2)
                scale = _soften_scale(scale, rsf_mask_v[iv, jN], η_tan_v[iv, jN], η_prev_v[iv, jN], γrow, γ_ratio, scale_min)
            end
        end
        _apply_dτ_scale!(dτVx, αVx, βVx, cVx, i, j, scale)
    end
    if i ≤ size(dτVy, 1) && j ≤ size(dτVy, 2)
        scale = one(eltype(dτVy))
        jv = j + 1
        if jv ≤ size(rsf_mask_v, 2)
            γS = γ_eff[min(i, size(γ_eff, 1)), min(j, size(γ_eff, 2))]
            jN = min(j + 1, size(γ_eff, 2))
            γN = γ_eff[min(i, size(γ_eff, 1)), jN]
            γrow = max(γS, γN)
            if i ≤ size(rsf_mask_v, 1)
                scale = _soften_scale(scale, rsf_mask_v[i, jv], η_tan_v[i, jv], η_prev_v[i, jv], γrow, γ_ratio, scale_min)
            end
            ie = i + 1
            if ie ≤ size(rsf_mask_v, 1)
                scale = _soften_scale(scale, rsf_mask_v[ie, jv], η_tan_v[ie, jv], η_prev_v[ie, jv], γrow, γ_ratio, scale_min)
            end
        end
        _apply_dτ_scale!(dτVy, αVy, βVy, cVy, i, j, scale)
    end
    return nothing
end

@inline function _soften_scale(scale, mask, ηn, ηo, γrow, γ_ratio, scale_min = 0.1)
    if mask > 0.5 && isfinite(ηn) && isfinite(ηo) && ηn > 0 && ηo > ηn && γrow > γ_ratio * ηn
        return min(scale, max(sqrt(ηn / ηo), scale_min))
    end
    return scale
end

@inline function _apply_dτ_scale!(dτV, αV, βV, cV, i, j, scale)
    if scale < 1
        dτ = dτV[i, j] * scale
        dτV[i, j] = dτ
        cdt = cV[i, j] * dτ
        inv_den = inv(2 + cdt)
        βV[i, j] = 2 * dτ * inv_den
        αV[i, j] = (2 - cdt) * inv_den
    end
    return nothing
end

# The inner residual uses the secant viscosity. Shorten dτ where the frozen-state
# slope is softer than that secant, by √(η_tan/η_ve). Never lengthen the step.
@inline function _tangent_step_scale(scale, mask, ηij, ηtan, phase, rheology, dt)
    if mask > 0.5 && isfinite(ηtan) && ηtan > 0 && isfinite(ηij) && ηij > 0
        G = fn_ratio(get_shear_modulus, rheology, phase)
        ηve = ηij / @muladd(1 + ηij * inv(G * dt))
        if isfinite(ηve) && ηtan < ηve
            return min(scale, max(sqrt(ηtan / ηve), 1.0e-4))
        end
    end
    return scale
end

@parallel_indices (i, j) function _scale_dτ_by_tangent!(
        dτVx, dτVy, αVx, αVy, βVx, βVy, cVx, cVy,
        ηv, η_tan_v, mask_v, phase_vertex, rheology, dt,
    )
    if i ≤ size(dτVx, 1) && j ≤ size(dτVx, 2)
        scale = one(eltype(dτVx))
        iv = i + 1
        if iv ≤ size(mask_v, 1)
            if j ≤ size(mask_v, 2)
                scale = _tangent_step_scale(scale, mask_v[iv, j], ηv[iv, j], η_tan_v[iv, j], phase_vertex[iv, j], rheology, dt)
            end
            jn = j + 1
            if jn ≤ size(mask_v, 2)
                scale = _tangent_step_scale(scale, mask_v[iv, jn], ηv[iv, jn], η_tan_v[iv, jn], phase_vertex[iv, jn], rheology, dt)
            end
        end
        _apply_dτ_scale!(dτVx, αVx, βVx, cVx, i, j, scale)
    end
    if i ≤ size(dτVy, 1) && j ≤ size(dτVy, 2)
        scale = one(eltype(dτVy))
        jv = j + 1
        if jv ≤ size(mask_v, 2)
            if i ≤ size(mask_v, 1)
                scale = _tangent_step_scale(scale, mask_v[i, jv], ηv[i, jv], η_tan_v[i, jv], phase_vertex[i, jv], rheology, dt)
            end
            ie = i + 1
            if ie ≤ size(mask_v, 1)
                scale = _tangent_step_scale(scale, mask_v[ie, jv], ηv[ie, jv], η_tan_v[ie, jv], phase_vertex[ie, jv], rheology, dt)
            end
        end
        _apply_dτ_scale!(dτVy, αVy, βVy, cVy, i, j, scale)
    end
    return nothing
end

function scale_dτ_by_tangent!(dyrel, stokes, phase_ratios, rheology, dt, η_tan_v, rsf_mask_v)
    ni = covering_size((dyrel.dτVx, dyrel.dτVy))
    @parallel (@idx ni) _scale_dτ_by_tangent!(
        dyrel.dτVx, dyrel.dτVy, dyrel.αVx, dyrel.αVy, dyrel.βVx, dyrel.βVy,
        dyrel.cVx, dyrel.cVy, stokes.viscosity.ηv, η_tan_v, rsf_mask_v,
        phase_ratios.vertex, rheology, dt,
    )
    return nothing
end

@parallel_indices (i, j) function _store_tangent_scale!(scale, ηv, η_tan_v, mask_v, phase_vertex, rheology, dt)
    scale[i, j] = _tangent_step_scale(
        one(eltype(scale)), mask_v[i, j], ηv[i, j], η_tan_v[i, j], phase_vertex[i, j], rheology, dt,
    )
    return nothing
end

@parallel_indices (i, j) function _masked_finite!(out, A, mask, empty)
    a = A[i, j]
    out[i, j] = (mask[i, j] > 0.5 && isfinite(a)) ? a : empty
    return nothing
end

@parallel_indices (i, j) function _masked_abs!(out, A, mask, empty)
    a = A[i, j]
    out[i, j] = (mask[i, j] > 0.5 && isfinite(a)) ? abs(a) : empty
    return nothing
end

function shrink_dτ_where_penalty_dominates!(dyrel, η_tan_v, η_prev_v, rsf_mask_v, γ_ratio, scale_min = 0.1)
    ni = covering_size((dyrel.dτVx, dyrel.dτVy))
    @parallel (@idx ni) _shrink_dτ_rsf!(
        dyrel.dτVx, dyrel.dτVy, dyrel.αVx, dyrel.αVy, dyrel.βVx, dyrel.βVy,
        dyrel.cVx, dyrel.cVy, dyrel.γ_eff, η_tan_v, η_prev_v, rsf_mask_v, γ_ratio, scale_min,
    )
    return nothing
end

function Gershgorin_Stokes3D_SchurComplement!(Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
    ni = size(η)
    @parallel (@idx ni) _Gershgorin_Stokes3D_SchurComplement!(
        Dx,
        Dy,
        Dz,
        λmaxVx,
        λmaxVy,
        λmaxVz,
        η,
        γ_eff,
        di.center,
        di.vertex,
        phase_ratios.center,
        phase_ratios.yz,
        phase_ratios.xz,
        phase_ratios.xy,
        rheology,
        dt,
    )
    return nothing
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{2}, Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt
    )
    return Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{2}, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt
    )
    return Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{3}, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt
    )
    return Gershgorin_Stokes3D_SchurComplement!(Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
end

Base.@propagate_inbounds @inline function _ηve(ηij, rheology, phase, dt)
    Gij = fn_ratio(get_shear_modulus, rheology, phase)
    return ηij / @muladd(1 + ηij * inv(Gij * dt))
end

Base.@propagate_inbounds @inline _ηve_center(η, phase_center, rheology, dt, i, j, k) =
    _ηve(η[i, j, k], rheology, phase_center[i, j, k], dt)

# `periodic` is threaded through so these averages use the same wrapped cell stencil as the stress
# kernel does; the Gershgorin bound only holds for the operator it is actually built from.
Base.@propagate_inbounds @inline function _ηve_yz(η, phase_yz, rheology, dt, ni, periodic, i, j, k)
    Ic = clamped_indices(ni, periodic, i, j, k)
    return _ηve(harm_clamped_yz(η, Ic...), rheology, phase_yz[i, j, k], dt)
end

Base.@propagate_inbounds @inline function _ηve_xz(η, phase_xz, rheology, dt, ni, periodic, i, j, k)
    Ic = clamped_indices(ni, periodic, i, j, k)
    return _ηve(harm_clamped_xz(η, Ic...), rheology, phase_xz[i, j, k], dt)
end

Base.@propagate_inbounds @inline function _ηve_xy(η, phase_xy, rheology, dt, ni, periodic, i, j, k)
    Ic = clamped_indices(ni, periodic, i, j, k)
    return _ηve(harm_clamped_xy(η, Ic...), rheology, phase_xy[i, j, k], dt)
end

@parallel_indices (i, j, k) function _Gershgorin_Stokes3D_SchurComplement!(
        Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, γ_eff, di_center, di_vertex,
        phase_center, phase_yz, phase_xz, phase_xy, rheology, dt
    )

    ni = size(η)
    # A direction is periodic exactly when its momentum block carries a row per cell rather than
    # one per interior face, so the shapes of `Dx`/`Dy`/`Dz` are the flags -- no extra argument.
    periodic = (size(Dx, 1) == ni[1], size(Dy, 2) == ni[2], size(Dz, 3) == ni[3])
    c23 = 2 / 3
    c43 = 4 / 3
    ηC = _ηve_center(η, phase_center, rheology, dt, i, j, k)
    γC = γ_eff[i, j, k]

    # DYREL D/λ arrays store active velocity updates; boundary values are enforced by flow_bcs! after the shifted update.
    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2) && k ≤ size(Dx, 3)
        _dx = inv(@dx(di_center, i))
        _dy = inv(@dy(di_vertex, j))
        _dz = inv(@dz(di_vertex, k))
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dz2 = _dz * _dz
        _dxdy = _dx * _dy
        _dxdz = _dx * _dz

        # `Dx` reaches `i = ni[1]` only when the x-direction is periodic; that row belongs to the
        # seam face, whose eastern cell is the first one. The shear reads above are already right
        # there -- index `ni[1] + 1` is the seam plane of the `xy`/`xz` vertex arrays.
        iE = wrap_next(i, ni[1])
        ηW = ηC
        ηE = _ηve_center(η, phase_center, rheology, dt, iE, j, k)
        ηS = _ηve_xy(η, phase_xy, rheology, dt, ni, periodic, i + 1, j, k)
        ηN = _ηve_xy(η, phase_xy, rheology, dt, ni, periodic, i + 1, j + 1, k)
        ηB = _ηve_xz(η, phase_xz, rheology, dt, ni, periodic, i + 1, j, k)
        ηF = _ηve_xz(η, phase_xz, rheology, dt, ni, periodic, i + 1, j, k + 1)
        γW = γC
        γE = γ_eff[iE, j, k]
        γηW = γW + c43 * ηW
        γηE = γE + c43 * ηE
        γτW = γW - c23 * ηW
        γτE = γE - c23 * ηE

        Dx_ijk = Dx[i, j, k] = @muladd(
            (ηN + ηS) * _dy2 +
                (ηB + ηF) * _dz2 +
                (γηE + γηW) * _dx2
        )

        # ηve and γ_eff are nonnegative, so only the mixed-component coefficients can change sign.
        Cx = @muladd(
            (γηE + γηW) * _dx2 +
                (ηN + ηS) * _dy2 +
                (ηB + ηF) * _dz2 +
                abs((γτE + ηN) * _dxdy) +
                abs((γτE + ηS) * _dxdy) +
                abs((γτW + ηN) * _dxdy) +
                abs((γτW + ηS) * _dxdy) +
                abs((γτE + ηB) * _dxdz) +
                abs((γτW + ηB) * _dxdz) +
                abs((γτE + ηF) * _dxdz) +
                abs((γτW + ηF) * _dxdz) +
                Dx_ijk
        )

        λmaxVx[i, j, k] = Cx * inv(Dx_ijk)
    end

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2) && k ≤ size(Dy, 3)
        _dx = inv(@dx(di_vertex, i))
        _dy = inv(@dy(di_center, j))
        _dz = inv(@dz(di_vertex, k))
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dz2 = _dz * _dz
        _dxdy = _dx * _dy
        _dydz = _dy * _dz

        # Counterpart of `iE` in the x-block: the northern cell of the y seam face is the first.
        jN = wrap_next(j, ni[2])
        ηW = _ηve_xy(η, phase_xy, rheology, dt, ni, periodic, i, j + 1, k)
        ηE = _ηve_xy(η, phase_xy, rheology, dt, ni, periodic, i + 1, j + 1, k)
        ηS = ηC
        ηN = _ηve_center(η, phase_center, rheology, dt, i, jN, k)
        ηB = _ηve_yz(η, phase_yz, rheology, dt, ni, periodic, i, j + 1, k)
        ηF = _ηve_yz(η, phase_yz, rheology, dt, ni, periodic, i, j + 1, k + 1)
        γS = γC
        γN = γ_eff[i, jN, k]
        γηS = γS + c43 * ηS
        γηN = γN + c43 * ηN
        γτS = γS - c23 * ηS
        γτN = γN - c23 * ηN

        Dy_ijk = Dy[i, j, k] = @muladd(
            (ηE + ηW) * _dx2 +
                (ηB + ηF) * _dz2 +
                (γηN + γηS) * _dy2
        )

        Cy = @muladd(
            (ηE + ηW) * _dx2 +
                (γηN + γηS) * _dy2 +
                (ηB + ηF) * _dz2 +
                abs((γτN + ηE) * _dxdy) +
                abs((γτS + ηE) * _dxdy) +
                abs((γτN + ηW) * _dxdy) +
                abs((γτS + ηW) * _dxdy) +
                abs((γτN + ηB) * _dydz) +
                abs((γτS + ηB) * _dydz) +
                abs((γτN + ηF) * _dydz) +
                abs((γτS + ηF) * _dydz) +
                Dy_ijk
        )

        λmaxVy[i, j, k] = Cy * inv(Dy_ijk)
    end

    if i ≤ size(Dz, 1) && j ≤ size(Dz, 2) && k ≤ size(Dz, 3)
        _dx = inv(@dx(di_vertex, i))
        _dy = inv(@dy(di_vertex, j))
        _dz = inv(@dz(di_center, k))
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dz2 = _dz * _dz
        _dxdz = _dx * _dz
        _dydz = _dy * _dz

        # Counterpart of `iE` in the x-block: the front cell of the z seam face is the first.
        kF = wrap_next(k, ni[3])
        ηW = _ηve_xz(η, phase_xz, rheology, dt, ni, periodic, i, j, k + 1)
        ηE = _ηve_xz(η, phase_xz, rheology, dt, ni, periodic, i + 1, j, k + 1)
        ηS = _ηve_yz(η, phase_yz, rheology, dt, ni, periodic, i, j, k + 1)
        ηN = _ηve_yz(η, phase_yz, rheology, dt, ni, periodic, i, j + 1, k + 1)
        ηB = ηC
        ηF = _ηve_center(η, phase_center, rheology, dt, i, j, kF)
        γB = γC
        γF = γ_eff[i, j, kF]
        γηB = γB + c43 * ηB
        γηF = γF + c43 * ηF
        γτB = γB - c23 * ηB
        γτF = γF - c23 * ηF

        Dz_ijk = Dz[i, j, k] = @muladd(
            (ηE + ηW) * _dx2 +
                (ηN + ηS) * _dy2 +
                (γηB + γηF) * _dz2
        )

        Cz = @muladd(
            (ηE + ηW) * _dx2 +
                (ηN + ηS) * _dy2 +
                (γηB + γηF) * _dz2 +
                abs((γτB + ηE) * _dxdz) +
                abs((γτB + ηW) * _dxdz) +
                abs((γτF + ηE) * _dxdz) +
                abs((γτF + ηW) * _dxdz) +
                abs((γτB + ηN) * _dydz) +
                abs((γτB + ηS) * _dydz) +
                abs((γτF + ηN) * _dydz) +
                abs((γτF + ηS) * _dydz) +
                Dz_ijk
        )

        λmaxVz[i, j, k] = Cz * inv(Dz_ijk)
    end

    return nothing
end

# Smallest launch range covering every array in `A`. The per-face arrays have different shapes --
# and a periodic direction makes one of them a row longer -- so the range has to be taken over all
# of them; each kernel guards its own component against the excess.
@inline covering_size(A::NTuple{N, AbstractArray{T, N}}) where {N, T} =
    ntuple(d -> maximum(Aᵢ -> size(Aᵢ, d), A), Val(N))

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
    ni = covering_size(βV)
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
            cdt = cV_ij * dτV_ij
            inv_den = inv(2 + cdt)
            βV[i][I...] = @muladd 2 * dτV_ij * inv_den
            αV[i][I...] = @muladd (2 - cdt) * inv_den
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
    ni = covering_size(βV)
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
            # A non-positive or non-finite λmax would make dτ Inf or NaN and
            # poison the velocity on the next update. A zero step is safe.
            λ = λmaxV[i][I...]
            dτV_ij = dτV[i][I...] = (isfinite(λ) && λ > 0) ? 2 / √(λ) * CFL_v : zero(λ)
            cV_ij = cV[i][I...]
            cdt = cV_ij * dτV_ij
            inv_den = inv(2 + cdt)
            βV[i][I...] = @muladd 2 * dτV_ij * inv_den
            αV[i][I...] = @muladd (2 - cdt) * inv_den
        end
    end
    return nothing
end

function update_α_β!(dyrel::JustRelax.DYREL)
    return update_α_β!(Val(ndims(dyrel.γ_eff)), dyrel)
end

function update_α_β!(::Val{2}, dyrel::JustRelax.DYREL)
    return update_α_β!(
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.cVx, dyrel.cVy)
    )
end

function update_α_β!(::Val{3}, dyrel::JustRelax.DYREL)
    return update_α_β!(
        (dyrel.βVx, dyrel.βVy, dyrel.βVz),
        (dyrel.αVx, dyrel.αVy, dyrel.αVz),
        (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
        (dyrel.cVx, dyrel.cVy, dyrel.cVz)
    )
end

function update_dτV_α_β!(dyrel::JustRelax.DYREL)
    return update_dτV_α_β!(dyrel, dyrel.CFL)
end

function update_dτV_α_β!(dyrel::JustRelax.DYREL, CFL_v)
    return update_dτV_α_β!(Val(ndims(dyrel.γ_eff)), dyrel, CFL_v)
end

function update_dτV_α_β!(::Val{2}, dyrel::JustRelax.DYREL, CFL_v)
    return update_dτV_α_β!(
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.cVx, dyrel.cVy),
        (dyrel.λmaxVx, dyrel.λmaxVy),
        CFL_v
    )
end

function update_dτV_α_β!(::Val{3}, dyrel::JustRelax.DYREL, CFL_v)
    return update_dτV_α_β!(
        (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
        (dyrel.βVx, dyrel.βVy, dyrel.βVz),
        (dyrel.αVx, dyrel.αVy, dyrel.αVz),
        (dyrel.cVx, dyrel.cVy, dyrel.cVz),
        (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
        CFL_v
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
