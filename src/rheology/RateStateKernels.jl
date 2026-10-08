# ParallelStencil kernels for Rate-and-State Friction (2D centers + vertices).
# Vertices correspond to LaMEM XY edges / JR shear nodes (τ.xy, ηv, ε.xy).

# Host materialization: `Array(CellArray)` scalar-indexes on CUDA; Adapt bulk-copies `.data`.
@inline _host(A) = Adapt.adapt(Array, A)

"""
    init_rate_state_fields!(rsf_arr, ctrl, phase_ratios, xci, xvi=xci)

Initialize `Ω`/`Ω_old` (and vertex counterparts) from `state_rsf_init` of the dominant
phase, fill `a_eff`/`b_eff` (constant or spatial profile), and masks.
Host-side fill so spatial profile NamedTuples stay off-device.
"""
function init_rate_state_fields!(
        rsf_arr::JustRelax.RateStateArrays, ctrl::RateStateController, phase_ratios, xci, xvi = xci
    )
    ctrl.enabled || return nothing
    if rsf_do_center(rsf_arr)
        _init_rsf_location_host!(
            rsf_arr.Ω, rsf_arr.Ω_old, rsf_arr.a_eff, rsf_arr.b_eff, rsf_arr.rsf_mask,
            ctrl, phase_ratios.center, xci,
        )
    end
    if rsf_do_vertex(rsf_arr)
        _init_rsf_location_host!(
            rsf_arr.Ωv, rsf_arr.Ω_old_v, rsf_arr.a_eff_v, rsf_arr.b_eff_v, rsf_arr.rsf_mask_v,
            ctrl, phase_ratios.vertex, xvi,
        )
    end
    return nothing
end

function _init_rsf_location_host!(Ω_d, Ω_old_d, a_eff_d, b_eff_d, mask_d, ctrl, phase_loc, xi)
    phase = _host(phase_loc)
    ni = size(Ω_d)
    x = _host(xi[1])
    y = _host(xi[2])
    Ω = zeros(eltype(Ω_d), ni...)
    Ω_old = similar(Ω)
    a_eff = similar(Ω)
    b_eff = similar(Ω)
    mask = similar(Ω)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        ip = Int(argmax(phase[i, j]))
        p = ctrl.phases[ip]
        xi_i = x[clamp(i, 1, length(x))]
        yi_j = y[clamp(j, 1, length(y))]
        a, b = evaluate_rsf_ab(ctrl, ip, xi_i, yi_j)
        a_eff[i, j] = a
        b_eff[i, j] = b
        if p.enabled
            Ω[i, j] = p.state_init
            Ω_old[i, j] = p.state_init
            mask[i, j] = 1.0
        else
            Ω[i, j] = 0.0
            Ω_old[i, j] = 0.0
            mask[i, j] = 0.0
        end
    end
    copyto!(Ω_d, Ω)
    copyto!(Ω_old_d, Ω_old)
    copyto!(a_eff_d, a_eff)
    copyto!(b_eff_d, b_eff)
    copyto!(mask_d, mask)
    return nothing
end

"""
    refresh_rsf_ab_mask!(rsf_arr, ctrl, phase_ratios, xci, xvi=xci)

Recompute `a_eff`/`b_eff`/`rsf_mask` (and vertex counterparts) from current phase ratios.
Does not reset `Ω`.
"""
function refresh_rsf_ab_mask!(
        rsf_arr::JustRelax.RateStateArrays, ctrl::RateStateController, phase_ratios, xci, xvi = xci
    )
    ctrl.enabled || return nothing
    if rsf_do_center(rsf_arr)
        _refresh_rsf_ab_mask_host!(
            rsf_arr.a_eff, rsf_arr.b_eff, rsf_arr.rsf_mask,
            ctrl, phase_ratios.center, xci,
        )
    end
    if rsf_do_vertex(rsf_arr)
        _refresh_rsf_ab_mask_host!(
            rsf_arr.a_eff_v, rsf_arr.b_eff_v, rsf_arr.rsf_mask_v,
            ctrl, phase_ratios.vertex, xvi,
        )
    end
    return nothing
end

function _refresh_rsf_ab_mask_host!(a_eff_d, b_eff_d, mask_d, ctrl, phase_loc, xi)
    phase = _host(phase_loc)
    ni = size(a_eff_d)
    x = _host(xi[1])
    y = _host(xi[2])
    a_eff = _host(a_eff_d)
    b_eff = _host(b_eff_d)
    mask = _host(mask_d)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        ip = Int(argmax(phase[i, j]))
        p = ctrl.phases[ip]
        xi_i = x[clamp(i, 1, length(x))]
        yi_j = y[clamp(j, 1, length(y))]
        a, b = evaluate_rsf_ab(ctrl, ip, xi_i, yi_j)
        a_eff[i, j] = a
        b_eff[i, j] = b
        mask[i, j] = p.enabled ? 1.0 : 0.0
    end
    copyto!(a_eff_d, a_eff)
    copyto!(b_eff_d, b_eff)
    copyto!(mask_d, mask)
    return nothing
end

@inline function _dominant_phase(phase_ratio, i, j)
    return Int(argmax(@inbounds phase_ratio[i, j]))
end

"""
    apply_rsf_viscosity!(stokes, rsf_arr, ctrl, phase_ratios, dt; kwargs...)

LaMEM-style local constitutive solve with frozen `Ω_old` on centers and/or vertices
(`rsf_arr.loc`):

1. Viscoelastic trial rate ``DII`` (centers: ``ε + τ_o/(2Gdt)``; vertices: same with
   averaged normals + native ``ε.xy``, ``τ_o.xy``).
2. Bisection on effective ``η`` so ``DII = DIIels + DIIcreep + DIIrsf``.
3. Write the equivalent Maxwell creep buffer onto `η` (centers) and/or `ηv` (vertices).

`η` / `ηv` on entry must be the GeoParams / Maxwell creep buffers (restored each PT iter
when stacking must be avoided).
"""
function apply_rsf_viscosity!(
        stokes,
        rsf_arr::JustRelax.RateStateArrays,
        ctrl::RateStateController,
        phase_ratios,
        dt;
        ε_floor = 1.0e-30,
        P_floor = 1.0e3,
        viscosity_cutoff = (-Inf, Inf),
        tol_rel = 1.0e-12,
        maxit = 50,
    )
    ctrl.enabled || return nothing
    η_min, η_max = viscosity_cutoff
    Gdt = ctrl.G * dt
    if rsf_do_center(rsf_arr)
        ni = size(stokes.P)
        @parallel (@idx ni) _apply_rsf_viscosity_center!(
            stokes.viscosity.η,
            stokes.ε.xx,
            stokes.ε.yy,
            stokes.ε.xy_c,
            stokes.τ_o.xx,
            stokes.τ_o.yy,
            stokes.τ_o.xy_c,
            stokes.P,
            rsf_arr.Ω_old,
            rsf_arr.a_eff,
            rsf_arr.b_eff,
            rsf_arr.Vp,
            rsf_arr.τ_rsf,
            rsf_arr.rsf_mask,
            phase_ratios.center,
            ctrl.phases,
            Gdt,
            ε_floor,
            P_floor,
            η_min,
            η_max,
            tol_rel,
            maxit,
        )
    end
    if rsf_do_vertex(rsf_arr)
        nv = size(stokes.viscosity.ηv)
        periodic = periodic_dims(stokes)
        @parallel (@idx nv) _apply_rsf_viscosity_vertex!(
            stokes.viscosity.ηv,
            stokes.ε.xx,
            stokes.ε.yy,
            stokes.ε.xy,
            stokes.τ_o.xx_v,
            stokes.τ_o.yy_v,
            stokes.τ_o.xy,
            stokes.P,
            rsf_arr.Ω_old_v,
            rsf_arr.a_eff_v,
            rsf_arr.b_eff_v,
            rsf_arr.Vp_v,
            rsf_arr.τ_rsf_v,
            rsf_arr.rsf_mask_v,
            phase_ratios.vertex,
            ctrl.phases,
            Gdt,
            ε_floor,
            P_floor,
            η_min,
            η_max,
            tol_rel,
            maxit,
            periodic,
        )
    end
    return nothing
end

@parallel_indices (i, j) function _apply_rsf_viscosity_center!(
        η, εxx, εyy, εxy_c, τxx_o, τyy_o, τxy_o, P,
        Ω_old, a_eff, b_eff, Vp, τ_rsf_arr, rsf_mask, phase_center, phases,
        Gdt, ε_floor, P_floor, η_min, η_max, tol_rel, maxit,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_center, i, j)
        p = phases[ip]
        if p.enabled
            _Gdt = inv(Gdt)
            εxx_ve = εxx[i, j] + 0.5 * τxx_o[i, j] * _Gdt
            εyy_ve = εyy[i, j] + 0.5 * τyy_o[i, j] * _Gdt
            εxy_ve = εxy_c[i, j] + 0.5 * τxy_o[i, j] * _Gdt
            DII = sqrt(0.5 * (εxx_ve^2 + εyy_ve^2) + εxy_ve^2)
            DII = max(DII, ε_floor)
            Pij = max(abs(P[i, j]), P_floor)
            r = rsf_from_phase_params(p, a_eff[i, j], b_eff[i, j])
            η_creep = η[i, j]
            _, η_creep_out, Vp_ij, _ = rsf_effective_viscosity(
                r;
                DII = DII,
                η_creep = η_creep,
                Gdt = Gdt,
                Ω_old = Ω_old[i, j],
                P = Pij,
                tol_rel = tol_rel,
                maxit = maxit,
                ε_floor = ε_floor,
            )
            τ_rsf_arr[i, j] = compute_stress_frozen_Ω(r; ε = DII, Ω_old = Ω_old[i, j], P = Pij)
            Vp[i, j] = _clamp_Vp(Vp_ij)
            η_new = min(max(η_creep_out, η_min), η_max)
            if isfinite(η_new) && η_new > ε_floor
                η[i, j] = η_new
            end
        end
    end
    return nothing
end

@parallel_indices (i, j) function _apply_rsf_viscosity_vertex!(
        ηv, εxx, εyy, εxy, τxx_ov, τyy_ov, τxy_o, P,
        Ω_old, a_eff, b_eff, Vp, τ_rsf_arr, rsf_mask, phase_vertex, phases,
        Gdt, ε_floor, P_floor, η_min, η_max, tol_rel, maxit, periodic,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_vertex, i, j)
        p = phases[ip]
        if p.enabled
            ni = size(εxx)
            Ic = clamped_indices(ni, periodic, i, j)
            _Gdt = inv(Gdt)
            εxx_ve = av_clamped(εxx, Ic...) + 0.5 * τxx_ov[i, j] * _Gdt
            εyy_ve = av_clamped(εyy, Ic...) + 0.5 * τyy_ov[i, j] * _Gdt
            εxy_ve = εxy[i, j] + 0.5 * τxy_o[i, j] * _Gdt
            DII = sqrt(0.5 * (εxx_ve^2 + εyy_ve^2) + εxy_ve^2)
            DII = max(DII, ε_floor)
            Pij = max(abs(av_clamped(P, Ic...)), P_floor)
            r = rsf_from_phase_params(p, a_eff[i, j], b_eff[i, j])
            η_creep = ηv[i, j]
            _, η_creep_out, Vp_ij, _ = rsf_effective_viscosity(
                r;
                DII = DII,
                η_creep = η_creep,
                Gdt = Gdt,
                Ω_old = Ω_old[i, j],
                P = Pij,
                tol_rel = tol_rel,
                maxit = maxit,
                ε_floor = ε_floor,
            )
            τ_rsf_arr[i, j] = compute_stress_frozen_Ω(r; ε = DII, Ω_old = Ω_old[i, j], P = Pij)
            Vp[i, j] = _clamp_Vp(Vp_ij)
            η_new = min(max(η_creep_out, η_min), η_max)
            if isfinite(η_new) && η_new > ε_floor
                ηv[i, j] = η_new
            end
        end
    end
    return nothing
end

"""
Hard-clamp slip rate to `[-Vp_max_abs, Vp_max_abs]` (default 100 m/s).
Non-finite values map to `+Vp_max_abs`.
"""
@inline function _clamp_Vp(v, Vp_max_abs = 100.0)
    if !(isfinite(v))
        return oftype(v, Vp_max_abs)
    elseif abs(v) > Vp_max_abs
        return copysign(oftype(v, Vp_max_abs), v)
    else
        return v
    end
end

"""
    check_Vp_rsf!(rsf_arr; Vp_max_abs=100.0)

Hard-clamp masked `Vp` / `Vp_v` to `[-Vp_max_abs, Vp_max_abs]` (default 100 m/s).
Non-finite values are replaced by `+Vp_max_abs`.
"""
function check_Vp_rsf!(rsf_arr::JustRelax.RateStateArrays; Vp_max_abs = 100.0)
    if rsf_do_center(rsf_arr)
        @parallel (@idx size(rsf_arr.Vp)) _clamp_Vp_rsf_field!(
            rsf_arr.Vp, rsf_arr.rsf_mask, Vp_max_abs
        )
    end
    if rsf_do_vertex(rsf_arr)
        @parallel (@idx size(rsf_arr.Vp_v)) _clamp_Vp_rsf_field!(
            rsf_arr.Vp_v, rsf_arr.rsf_mask_v, Vp_max_abs
        )
    end
    return nothing
end

@parallel_indices (I...) function _clamp_Vp_rsf_field!(Vp, mask, Vp_max_abs)
    @inbounds if mask[I...] > 0.5
        Vp[I...] = _clamp_Vp(Vp[I...], Vp_max_abs)
    end
    return nothing
end

"""
    enforce_rsf_stress!(stokes, rsf_arr, ctrl)

Scale center stress so ``τII = τ_rsf`` on RSF cells (diagnostic / optional closure).
Call after the APT stress update; then refresh vertex ``τ.xy`` via `center2vertex!`.
"""
function enforce_rsf_stress!(stokes, rsf_arr::JustRelax.RateStateArrays, ctrl::RateStateController)
    ctrl.enabled || return nothing
    rsf_do_center(rsf_arr) || return nothing
    ni = size(stokes.P)
    @parallel (@idx ni) _enforce_rsf_stress!(
        stokes.τ.xx,
        stokes.τ.yy,
        stokes.τ.xy_c,
        stokes.τ.II,
        rsf_arr.τ_rsf,
        rsf_arr.rsf_mask,
    )
    center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
    return nothing
end

@parallel_indices (i, j) function _enforce_rsf_stress!(τxx, τyy, τxy_c, τII, τ_rsf, rsf_mask)
    @inbounds if rsf_mask[i, j] > 0.5
        τ_target = τ_rsf[i, j]
        τ_now = τII[i, j]
        if τ_target > 0 && τ_now > 1.0e-30 && isfinite(τ_target)
            f = τ_target / τ_now
            τxx[i, j] *= f
            τyy[i, j] *= f
            τxy_c[i, j] *= f
            τII[i, j] = τ_target
        end
    end
    return nothing
end

"""
    update_rate_state!(rsf_arr, stokes, ctrl, phase_ratios, dt)

After APT convergence: LaMEM-style state update on centers and/or vertices.
`Vp` from stress with frozen `Ω_old`, then `Ω` advanced and copied to `Ω_old`.
"""
function update_rate_state!(
        rsf_arr::JustRelax.RateStateArrays, stokes, ctrl::RateStateController, phase_ratios, dt;
        Vp_max_abs = 100.0,
    )
    ctrl.enabled || return nothing
    if rsf_do_center(rsf_arr)
        shear2center!(stokes.ε)
        ni = size(stokes.P)
        @parallel (@idx ni) _update_rate_state_center!(
            rsf_arr.Ω,
            rsf_arr.Ω_old,
            rsf_arr.Vp,
            rsf_arr.a_eff,
            rsf_arr.b_eff,
            rsf_arr.rsf_mask,
            stokes.τ.II,
            stokes.P,
            phase_ratios.center,
            ctrl.phases,
            dt,
        )
    end
    if rsf_do_vertex(rsf_arr)
        nv = size(rsf_arr.Ωv)
        periodic = periodic_dims(stokes)
        @parallel (@idx nv) _update_rate_state_vertex!(
            rsf_arr.Ωv,
            rsf_arr.Ω_old_v,
            rsf_arr.Vp_v,
            rsf_arr.a_eff_v,
            rsf_arr.b_eff_v,
            rsf_arr.rsf_mask_v,
            stokes.τ.xx_v,
            stokes.τ.yy_v,
            stokes.τ.xy,
            stokes.P,
            phase_ratios.vertex,
            ctrl.phases,
            dt,
            periodic,
        )
    end
    check_Vp_rsf!(rsf_arr; Vp_max_abs = Vp_max_abs)
    return nothing
end

@parallel_indices (i, j) function _update_rate_state_center!(
        Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, τII, P, phase_center, phases, dt
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_center, i, j)
        p = phases[ip]
        if p.enabled
            r = rsf_from_phase_params(p, a_eff[i, j], b_eff[i, j])
            Pij = max(abs(P[i, j]), 1.0e3)
            Vp_ij = _clamp_Vp(compute_Vp_from_stress(r; τ = τII[i, j], Ω = Ω_old[i, j], P = Pij))
            Ωnew = update_Ω(r; Ω_old = Ω_old[i, j], dt = dt, Vp = Vp_ij)
            Ω[i, j] = Ωnew
            Ω_old[i, j] = Ωnew
            Vp[i, j] = Vp_ij
        end
    end
    return nothing
end

@parallel_indices (i, j) function _update_rate_state_vertex!(
        Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, τxx_v, τyy_v, τxy, P, phase_vertex, phases, dt, periodic
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_vertex, i, j)
        p = phases[ip]
        if p.enabled
            r = rsf_from_phase_params(p, a_eff[i, j], b_eff[i, j])
            ni = size(P)
            Ic = clamped_indices(ni, periodic, i, j)
            Pij = max(abs(av_clamped(P, Ic...)), 1.0e3)
            τII = sqrt(0.5 * (τxx_v[i, j]^2 + τyy_v[i, j]^2) + τxy[i, j]^2)
            Vp_ij = _clamp_Vp(compute_Vp_from_stress(r; τ = τII, Ω = Ω_old[i, j], P = Pij))
            Ωnew = update_Ω(r; Ω_old = Ω_old[i, j], dt = dt, Vp = Vp_ij)
            Ω[i, j] = Ωnew
            Ω_old[i, j] = Ωnew
            Vp[i, j] = Vp_ij
        end
    end
    return nothing
end

"""
Host-side RSF adaptive dt (LaMEM `updatePhaseRSFState` + global min).

Returns `(dt_rsf, Vp_max, dt_h, dt_w, dt_c)`.
Uses stress-based `Vp` already stored on the grid after `update_rate_state!`.
"""
function compute_dt_ratestate_grid(
        rsf_arr::JustRelax.RateStateArrays,
        stokes,
        ctrl::RateStateController,
        phase_ratios,
        dt;
        kwargs...,
    )
    empty = (; dt_rsf = Inf, Vp_max = 0.0, dt_h = Inf, dt_w = Inf, dt_c = Inf)
    ctrl.enabled || return empty
    dt_rsf = typemax(Float64)
    dt_h_min = typemax(Float64)
    dt_w_min = typemax(Float64)
    dt_c_min = typemax(Float64)
    Vp_max = 0.0
    any_cell = false
    G, ν = ctrl.G, ctrl.ν

    if rsf_do_center(rsf_arr)
        dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell = _accumulate_dt_rsf_host!(
            dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell,
            _host(rsf_arr.Ω), _host(rsf_arr.a_eff), _host(rsf_arr.b_eff),
            _host(rsf_arr.rsf_mask), _host(rsf_arr.Vp), _host(phase_ratios.center),
            _host(stokes.P), ctrl, G, ν,
        )
    end
    if rsf_do_vertex(rsf_arr)
        # Pressure at vertices: simple cell-neighbor average on host
        P_c = _host(stokes.P)
        nv = size(rsf_arr.rsf_mask_v)
        P_v = zeros(eltype(P_c), nv...)
        nx, ny = size(P_c)
        @inbounds for j in 1:nv[2], i in 1:nv[1]
            i0 = max(i - 1, 1)
            ic = min(i, nx)
            j0 = max(j - 1, 1)
            jc = min(j, ny)
            P_v[i, j] = 0.25 * (P_c[i0, j0] + P_c[ic, jc] + P_c[i0, jc] + P_c[ic, j0])
        end
        dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell = _accumulate_dt_rsf_host!(
            dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell,
            _host(rsf_arr.Ωv), _host(rsf_arr.a_eff_v), _host(rsf_arr.b_eff_v),
            _host(rsf_arr.rsf_mask_v), _host(rsf_arr.Vp_v), _host(phase_ratios.vertex),
            P_v, ctrl, G, ν,
        )
    end
    any_cell || return empty
    dt_rsf = minimum_mpi(dt_rsf)
    Vp_max = maximum_mpi(Vp_max)
    return (; dt_rsf = dt_rsf, Vp_max = Vp_max, dt_h = dt_h_min, dt_w = dt_w_min, dt_c = dt_c_min)
end

function _accumulate_dt_rsf_host!(
        dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell,
        Ω, a_eff, b_eff, mask, Vp_arr, phase_c, P, ctrl, G, ν,
    )
    ni = size(mask)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        mask[i, j] ≤ 0.5 && continue
        ip = Int(argmax(phase_c[i, j]))
        p = ctrl.phases[ip]
        p.enabled || continue
        any_cell = true
        Vp_ij = Vp_arr[i, j]
        !(isfinite(Vp_ij) && Vp_ij > 0) && continue
        Vp_max = max(Vp_max, Vp_ij)
        Pij = max(abs(P[i, j]), 1.0e3)
        dti = compute_dt_ratestate_lamem(;
            Wf = p.Wf, L = p.L, V₀ = p.V₀,
            a = a_eff[i, j], b = b_eff[i, j],
            G = G, ν = ν, P = Pij, Vp = Vp_ij, Ω = Ω[i, j],
        )
        dth = dt_healing(; L = p.L, V₀ = p.V₀, Ω = Ω[i, j])
        dtw = dt_weakening(;
            L = p.L, Vp = Vp_ij,
            dteta_max = dteta_max_lamem(a_eff[i, j], b_eff[i, j], p.L, p.Wf, G, ν, Pij),
        )
        dtc = dt_courant(; Wf = p.Wf, Vp = Vp_ij)
        dt_h_min = min(dt_h_min, dth)
        dt_w_min = min(dt_w_min, dtw)
        dt_c_min = min(dt_c_min, dtc)
        dt_rsf = min(dt_rsf, dti)
    end
    return dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell
end
