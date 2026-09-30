# ParallelStencil kernels for Rate-and-State Friction (2D centers first).

"""
    init_rate_state_fields!(rsf_arr, ctrl, phase_ratios, xci)

Initialize `Ω`/`Ω_old` from `state_rsf_init` of the dominant phase, fill
`a_eff`/`b_eff` (constant or spatial profile), and `rsf_mask`.
Host-side fill so spatial profile NamedTuples stay off-device.
"""
function init_rate_state_fields!(rsf_arr::RateStateArrays, ctrl::RateStateController, phase_ratios, xci)
    ctrl.enabled || return nothing
    phase_c = Array(phase_ratios.center)  # Matrix of SVector ratios
    ni = size(rsf_arr.Ω)
    xc = Array(xci[1])
    yc = Array(xci[2])
    Ω = zeros(eltype(Array(rsf_arr.Ω)), ni...)
    Ω_old = similar(Ω)
    a_eff = similar(Ω)
    b_eff = similar(Ω)
    mask = similar(Ω)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        ip = Int(argmax(phase_c[i, j]))
        p = ctrl.phases[ip]
        x = xc[i]
        y = yc[j]
        a, b = evaluate_rsf_ab(ctrl, ip, x, y)
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
    copyto!(rsf_arr.Ω, Ω)
    copyto!(rsf_arr.Ω_old, Ω_old)
    copyto!(rsf_arr.a_eff, a_eff)
    copyto!(rsf_arr.b_eff, b_eff)
    copyto!(rsf_arr.rsf_mask, mask)
    return nothing
end

"""
    refresh_rsf_ab_mask!(rsf_arr, ctrl, phase_ratios, xci)

Recompute `a_eff`/`b_eff`/`rsf_mask` from current phase ratios (e.g. after advection).
Does not reset `Ω`.
"""
function refresh_rsf_ab_mask!(rsf_arr::RateStateArrays, ctrl::RateStateController, phase_ratios, xci)
    ctrl.enabled || return nothing
    phase_c = Array(phase_ratios.center)
    ni = size(rsf_arr.Ω)
    xc = Array(xci[1])
    yc = Array(xci[2])
    a_eff = Array(rsf_arr.a_eff)
    b_eff = Array(rsf_arr.b_eff)
    mask = Array(rsf_arr.rsf_mask)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        ip = Int(argmax(phase_c[i, j]))
        p = ctrl.phases[ip]
        a, b = evaluate_rsf_ab(ctrl, ip, xc[i], yc[j])
        a_eff[i, j] = a
        b_eff[i, j] = b
        mask[i, j] = p.enabled ? 1.0 : 0.0
    end
    copyto!(rsf_arr.a_eff, a_eff)
    copyto!(rsf_arr.b_eff, b_eff)
    copyto!(rsf_arr.rsf_mask, mask)
    return nothing
end

@inline function _dominant_phase(phase_center, i, j)
    return Int(argmax(@inbounds phase_center[i, j]))
end

"""
    apply_rsf_viscosity!(stokes, rsf_arr, ctrl, phase_ratios, dt; kwargs...)

With frozen `Ω_old`, compute RSF τII from center strain-rate invariant and
combine η with existing viscosity in series on RSF cells.
Call inside APT after strain rates + GeoParams viscosity; ensure `ε.xy_c` is current
(`shear2center!(stokes.ε)`).

`τ_rsf = P(1-λ)μ_d` — pressure must be meaningful (LaMEM uses `p_shift ≈ 5e6` when `g=0`).
"""
function apply_rsf_viscosity!(
        stokes,
        rsf_arr::RateStateArrays,
        ctrl::RateStateController,
        phase_ratios,
        dt;
        ε_floor = 1.0e-30,
        P_floor = 1.0e3,
        viscosity_cutoff = (-Inf, Inf),
    )
    ctrl.enabled || return nothing
    η_min, η_max = viscosity_cutoff
    ni = size(stokes.P)
    @parallel (@idx ni) _apply_rsf_viscosity!(
        stokes.viscosity.η,
        stokes.viscosity.η_vep,
        stokes.ε.xx,
        stokes.ε.yy,
        stokes.ε.xy_c,
        stokes.P,
        rsf_arr.Ω_old,
        rsf_arr.a_eff,
        rsf_arr.b_eff,
        rsf_arr.Vp,
        rsf_arr.rsf_mask,
        phase_ratios.center,
        ctrl.phases,
        ε_floor,
        P_floor,
        η_min,
        η_max,
    )
    return nothing
end

@parallel_indices (i, j) function _apply_rsf_viscosity!(
        η, η_vep, εxx, εyy, εxy_c, P, Ω_old, a_eff, b_eff, Vp, rsf_mask, phase_center, phases,
        ε_floor, P_floor, η_min, η_max,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_center, i, j)
        p = phases[ip]
        if p.enabled
            εxxij = εxx[i, j]
            εyyij = εyy[i, j]
            εxyij = εxy_c[i, j]
            εII = sqrt(0.5 * (εxxij^2 + εyyij^2) + εxyij^2)
            εII = max(εII, ε_floor)
            # Frictional strength needs effective normal stress; never let P → 0
            Pij = max(abs(P[i, j]), P_floor)
            r = RateStateFriction(p.λ, p.μ₀, p.V₀, a_eff[i, j], b_eff[i, j], p.L, p.C, p.Wf)
            τ_rsf = compute_stress_frozen_Ω(r; ε = εII, Ω_old = Ω_old[i, j], P = Pij)
            if isfinite(τ_rsf)
                Vp[i, j] = 2 * r.Wf * εII
                η_rsf = τ_rsf / (2 * εII)
                ηij = η[i, j]
                η_new = 1.0 / (1.0 / ηij + 1.0 / max(η_rsf, ε_floor))
                η_new = min(max(η_new, η_min), η_max)
                η[i, j] = η_new
                η_vep[i, j] = η_new
            end
        end
    end
    return nothing
end

"""
    update_rate_state!(rsf_arr, stokes, ctrl, phase_ratios, dt)

After APT convergence: LaMEM-style state update.
`Vp` is computed from **stress** (`compute_Vp_from_stress` with frozen `Ω_old`), then
`Ω` is advanced and copied to `Ω_old`.
"""
function update_rate_state!(
        rsf_arr::RateStateArrays, stokes, ctrl::RateStateController, phase_ratios, dt
    )
    ctrl.enabled || return nothing
    shear2center!(stokes.ε)
    ni = size(stokes.P)
    @parallel (@idx ni) _update_rate_state!(
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
    return nothing
end

@parallel_indices (i, j) function _update_rate_state!(
        Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, τII, P, phase_center, phases, dt
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_center, i, j)
        p = phases[ip]
        if p.enabled
            r = RateStateFriction(p.λ, p.μ₀, p.V₀, a_eff[i, j], b_eff[i, j], p.L, p.C, p.Wf)
            Pij = max(abs(P[i, j]), 1.0e3)
            # LaMEM: Vp from stress with state_old in the exp term
            Vp_ij = compute_Vp_from_stress(r; τ = τII[i, j], Ω = Ω_old[i, j], P = Pij)
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
Uses stress-based `Vp` already stored on the grid after `update_rate_state!`,
Lapusta ``dteta_max``, healing with **updated** Ω, slip Courant ``1e-3 Wf/Vp``.
"""
function compute_dt_ratestate_grid(
        rsf_arr::RateStateArrays,
        stokes,
        ctrl::RateStateController,
        phase_ratios,
        dt;
        kwargs...,
    )
    empty = (; dt_rsf = Inf, Vp_max = 0.0, dt_h = Inf, dt_w = Inf, dt_c = Inf)
    ctrl.enabled || return empty
    Ω = Array(rsf_arr.Ω)          # updated state (LaMEM dt_h uses *state)
    a_eff = Array(rsf_arr.a_eff)
    b_eff = Array(rsf_arr.b_eff)
    mask = Array(rsf_arr.rsf_mask)
    Vp_arr = Array(rsf_arr.Vp)
    phase_c = Array(phase_ratios.center)
    P = Array(stokes.P)
    dt_rsf = typemax(Float64)
    dt_h_min = typemax(Float64)
    dt_w_min = typemax(Float64)
    dt_c_min = typemax(Float64)
    Vp_max = 0.0
    ni = size(mask)
    any_cell = false
    G, ν = ctrl.G, ctrl.ν
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
    any_cell || return empty
    dt_rsf = try
        minimum_mpi(dt_rsf)
    catch
        dt_rsf
    end
    Vp_max = try
        maximum_mpi(Vp_max)
    catch
        Vp_max
    end
    return (; dt_rsf = dt_rsf, Vp_max = Vp_max, dt_h = dt_h_min, dt_w = dt_w_min, dt_c = dt_c_min)
end
