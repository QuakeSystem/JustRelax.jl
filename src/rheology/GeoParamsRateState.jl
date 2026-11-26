# GeoParams `RateStateFriction` on the staggered grid.
#
# The composite series solve (linear dashpot + friction + elasticity) owns the stress.
# This file only builds the trial invariant the Maxwell kernel already uses, writes the
# equivalent creep viscosity, and advances Ω after the step. It does not run the legacy
# bisection fold.

"""
    init_geoparams_rate_state!(fields, rheology, phase_ratios, state_init, xvi; xci=xvi, b_profiles=nothing)

Fill `Ω` / `Ω_old`, `a_eff` / `b_eff`, and the mask from the `GeoParams.RateStateFriction`
element of each phase. `state_init[phase]` is the starting `Ω`. `b_profiles[phase]` is
`nothing` or `(x_knots, values)` for a piecewise-linear `b(x)` (Herrendörfer phase 2).
Cells whose dominant phase has no rate-and-state element stay masked off.
"""
function init_geoparams_rate_state!(
        fields::JustRelax.RateStateArrays,
        rheology,
        phase_ratios,
        state_init,
        xvi;
        xci = xvi,
        b_profiles = nothing,
    )
    nphases = length(rheology)
    profiles = b_profiles === nothing ? ntuple(_ -> nothing, nphases) : b_profiles
    length(state_init) == nphases ||
        error("state_init must have one value per phase, got $(length(state_init)) for $nphases phases")
    length(profiles) == nphases ||
        error("b_profiles must have one entry per phase, got $(length(profiles)) for $nphases phases")
    if rsf_do_center(fields)
        _init_gp_rsf_location!(
            fields.Ω, fields.Ω_old, fields.a_eff, fields.b_eff, fields.rsf_mask,
            rheology, phase_ratios.center, xci, state_init, profiles,
        )
    end
    if rsf_do_vertex(fields)
        _init_gp_rsf_location!(
            fields.Ωv, fields.Ω_old_v, fields.a_eff_v, fields.b_eff_v, fields.rsf_mask_v,
            rheology, phase_ratios.vertex, xvi, state_init, profiles,
        )
    end
    return nothing
end

function _init_gp_rsf_location!(Ω_d, Ω_old_d, a_eff_d, b_eff_d, mask_d, rheology, phase_loc, xi, state_init, profiles)
    phase = Adapt.adapt(Array, phase_loc)
    ni = size(Ω_d)
    x = Adapt.adapt(Array, xi[1])
    Ω = zeros(eltype(Ω_d), ni...)
    Ω_old = zeros(eltype(Ω_d), ni...)
    a_eff = zeros(eltype(Ω_d), ni...)
    b_eff = zeros(eltype(Ω_d), ni...)
    mask = zeros(eltype(Ω_d), ni...)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        ip = Int(argmax(phase[i, j]))
        comp = rheology[ip].CompositeRheology[1]
        xi_i = x[clamp(i, 1, length(x))]
        if has_rate_state_friction(comp)
            r = rate_state_element(comp)
            Ω[i, j] = state_init[ip]
            Ω_old[i, j] = state_init[ip]
            a_eff[i, j] = float(r.a.val)
            b_eff[i, j] = _gp_b_of(r, profiles[ip], xi_i)
            mask[i, j] = 1.0
        end
    end
    copyto!(Ω_d, Ω)
    copyto!(Ω_old_d, Ω_old)
    copyto!(a_eff_d, a_eff)
    copyto!(b_eff_d, b_eff)
    copyto!(mask_d, mask)
    return nothing
end

@inline function _gp_b_of(r, profile, x)
    profile === nothing && return float(r.b.val)
    return get_rsf_profile(x, profile[1], profile[2])
end

"""Trial invariant matching the DYREL stress kernel: ``ε + τ_old / (2 G dt)``."""
@inline function _gp_trial_DII(comp, εxx, εyy, εxy, τxx, τyy, τxy, dt)
    G = shear_modulus_of(comp)
    inv2 = (G > 0 && dt > 0 && isfinite(dt)) ? inv(2 * G * dt) : 0.0
    return second_invariant(
        εxx + τxx * inv2,
        εyy + τyy * inv2,
        εxy + τxy * inv2,
    )
end

@generated function _gp_rsf_maxwell(
        rheology::NTuple{N, AbstractMaterialParamsStruct},
        ip::Int, εxx, εyy, εxy, τxx, τyy, τxy, P, Ω, a, b, dt,
    ) where {N}
    return quote
        Base.@_inline_meta
        Base.@nexprs $N i -> begin
            if ip == i
                comp = rheology[i].CompositeRheology[1]
                if has_rate_state_friction(comp)
                    DII = _gp_trial_DII(comp, εxx, εyy, εxy, τxx, τyy, τxy, dt)
                    T = typeof(float(DII))
                    args = (; P = P, Ω_old = Ω, a = a, b = b, dt = dt, τII_old = zero(T))
                    η = maxwell_creep_viscosity(comp, DII, args)
                    G = shear_modulus_of(comp)
                    Gdt = G * dt
                    η_ve = Gdt > 0 ? inv(inv(η) + inv(Gdt)) : η
                    τ = 2 * η_ve * max(DII, zero(T))
                    return (true, float(η), float(τ))
                else
                    return (false, 0.0, 0.0)
                end
            end
        end
        return (false, 0.0, 0.0)
    end
end

"""
    apply_geoparams_rsf_viscosity!(stokes, fields, phase_ratios, rheology, dt; viscosity_cutoff, P_floor=1e3)

One viscosity write. On masked points, `compute_τII` of the series composite supplies the
stress at the trial rate, and the stored creep viscosity is the one whose Maxwell update
reproduces that stress. `Ω_old` is read and not advanced.
"""
function apply_geoparams_rsf_viscosity!(
        stokes,
        fields::JustRelax.RateStateArrays,
        phase_ratios,
        rheology,
        dt;
        viscosity_cutoff = (-Inf, Inf),
        P_floor = 1.0e3,
    )
    ndims(stokes.P) == 2 ||
        error("GeoParams rate-and-state viscosity is implemented for 2D.")
    η_min, η_max = viscosity_cutoff
    if rsf_do_center(fields)
        shear2center!(stokes.ε)
        shear2center!(stokes.τ_o)
        ni = size(stokes.P)
        @parallel (@idx ni) _apply_gp_rsf_center!(
            stokes.viscosity.η,
            fields.τ_rsf,
            stokes.ε.xx, stokes.ε.yy, stokes.ε.xy_c,
            stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c,
            stokes.P,
            fields.Ω_old, fields.a_eff, fields.b_eff, fields.rsf_mask,
            phase_ratios.center, rheology,
            dt, P_floor, η_min, η_max,
        )
    end
    if rsf_do_vertex(fields)
        nv = size(stokes.viscosity.ηv)
        periodic = periodic_dims(stokes)
        @parallel (@idx nv) _apply_gp_rsf_vertex!(
            stokes.viscosity.ηv,
            fields.τ_rsf_v,
            stokes.ε.xx, stokes.ε.yy, stokes.ε.xy,
            stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy,
            stokes.P,
            fields.Ω_old_v, fields.a_eff_v, fields.b_eff_v, fields.rsf_mask_v,
            phase_ratios.vertex, rheology,
            dt, P_floor, η_min, η_max, periodic,
        )
    end
    update_halo!(stokes.viscosity.η, stokes.viscosity.ηv)
    return nothing
end

@parallel_indices (i, j) function _apply_gp_rsf_center!(
        η, τ_rsf, εxx, εyy, εxy, τxx, τyy, τxy, P,
        Ω_old, a_eff, b_eff, rsf_mask, phase_center, rheology,
        dt, P_floor, η_min, η_max,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_center, i, j)
        active, η_new, τ_new = _gp_rsf_maxwell(
            rheology, ip,
            εxx[i, j], εyy[i, j], εxy[i, j],
            τxx[i, j], τyy[i, j], τxy[i, j],
            max(abs(P[i, j]), P_floor),
            Ω_old[i, j], a_eff[i, j], b_eff[i, j], dt,
        )
        if active && isfinite(η_new) && η_new > 0
            η[i, j] = min(max(η_new, η_min), η_max)
            τ_rsf[i, j] = τ_new
        end
    end
    return nothing
end

@parallel_indices (i, j) function _apply_gp_rsf_vertex!(
        ηv, τ_rsf, εxx, εyy, εxy, τxx, τyy, τxy, P,
        Ω_old, a_eff, b_eff, rsf_mask, phase_vertex, rheology,
        dt, P_floor, η_min, η_max, periodic,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_vertex, i, j)
        ni = size(εxx)
        Ic = clamped_indices(ni, periodic, i, j)
        active, η_new, τ_new = _gp_rsf_maxwell(
            rheology, ip,
            av_clamped(εxx, Ic...), av_clamped(εyy, Ic...), εxy[i, j],
            τxx[i, j], τyy[i, j], τxy[i, j],
            max(abs(av_clamped(P, Ic...)), P_floor),
            Ω_old[i, j], a_eff[i, j], b_eff[i, j], dt,
        )
        if active && isfinite(η_new) && η_new > 0
            ηv[i, j] = min(max(η_new, η_min), η_max)
            τ_rsf[i, j] = τ_new
        end
    end
    return nothing
end

@generated function _gp_rsf_advance(
        rheology::NTuple{N, AbstractMaterialParamsStruct},
        ip::Int, τII, P, Ω_old, a, b, dt,
    ) where {N}
    return quote
        Base.@_inline_meta
        Base.@nexprs $N i -> begin
            if ip == i
                comp = rheology[i].CompositeRheology[1]
                if has_rate_state_friction(comp)
                    r = rate_state_element(comp)
                    ε = compute_εII(r, τII; P = P, Ω_old = Ω_old, a = a, b = b)
                    ε = float(first(ε))
                    ε = (isfinite(ε) && ε > 0) ? ε : 0.0
                    Vp = _clamp_Vp(2 * float(r.D.val) * ε)
                    Ωnew = GeoParams.update_Ω(r; ε = ε, Vp = Vp, Ω_old = Ω_old, dt = dt, a = a, b = b)
                    return (true, float(Vp), float(Ωnew))
                else
                    return (false, 0.0, float(Ω_old))
                end
            end
        end
        return (false, 0.0, float(Ω_old))
    end
end

"""
    update_geoparams_rate_state!(fields, stokes, phase_ratios, rheology, dt)

After a converged step. Slip rate is the frictional part of the series,
``Vp = 2 D \\varepsilon_{rsf}(\\tau)`` at frozen `Ω_old`, then the aging law advances `Ω`.
"""
function update_geoparams_rate_state!(
        fields::JustRelax.RateStateArrays, stokes, phase_ratios, rheology, dt;
        P_floor = 1.0e3,
    )
    ndims(stokes.P) == 2 ||
        error("GeoParams rate-and-state state update is implemented for 2D.")
    if rsf_do_center(fields)
        ni = size(stokes.P)
        @parallel (@idx ni) _update_gp_rsf_center!(
            fields.Ω, fields.Ω_old, fields.Vp,
            fields.a_eff, fields.b_eff, fields.rsf_mask,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c, stokes.P,
            phase_ratios.center, rheology, dt, P_floor,
        )
    end
    if rsf_do_vertex(fields)
        nv = size(fields.Ωv)
        periodic = periodic_dims(stokes)
        @parallel (@idx nv) _update_gp_rsf_vertex!(
            fields.Ωv, fields.Ω_old_v, fields.Vp_v,
            fields.a_eff_v, fields.b_eff_v, fields.rsf_mask_v,
            stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.P,
            phase_ratios.vertex, rheology, dt, P_floor, periodic,
        )
    end
    return nothing
end

@parallel_indices (i, j) function _update_gp_rsf_center!(
        Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, τxx, τyy, τxy, P,
        phase_center, rheology, dt, P_floor,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_center, i, j)
        τII = second_invariant(τxx[i, j], τyy[i, j], τxy[i, j])
        active, Vp_ij, Ωnew = _gp_rsf_advance(
            rheology, ip, τII, max(abs(P[i, j]), P_floor),
            Ω_old[i, j], a_eff[i, j], b_eff[i, j], dt,
        )
        if active && isfinite(Ωnew)
            Ω[i, j] = Ωnew
            Ω_old[i, j] = Ωnew
            Vp[i, j] = Vp_ij
        end
    end
    return nothing
end

@parallel_indices (i, j) function _update_gp_rsf_vertex!(
        Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, τxx, τyy, τxy, P,
        phase_vertex, rheology, dt, P_floor, periodic,
    )
    @inbounds if rsf_mask[i, j] > 0.5
        ip = _dominant_phase(phase_vertex, i, j)
        ni = size(P)
        Ic = clamped_indices(ni, periodic, i, j)
        τII = second_invariant(τxx[i, j], τyy[i, j], τxy[i, j])
        active, Vp_ij, Ωnew = _gp_rsf_advance(
            rheology, ip, τII, max(abs(av_clamped(P, Ic...)), P_floor),
            Ω_old[i, j], a_eff[i, j], b_eff[i, j], dt,
        )
        if active && isfinite(Ωnew)
            Ω[i, j] = Ωnew
            Ω_old[i, j] = Ωnew
            Vp[i, j] = Vp_ij
        end
    end
    return nothing
end

"""
    compute_dt_geoparams_rsf(fields, stokes, phase_ratios, rheology, dt) -> (; dt_rsf, Vp_max, dt_h, dt_w, dt_c)

LaMEM clamps (`θmax ∈ [0.1, 0.2]`) from the slip rate stored by
[`update_geoparams_rate_state!`](@ref). `Wf` is the law's fault width `D`.
"""
function compute_dt_geoparams_rsf(
        fields::JustRelax.RateStateArrays, stokes, phase_ratios, rheology, dt,
    )
    empty = (; dt_rsf = Inf, Vp_max = 0.0, dt_h = Inf, dt_w = Inf, dt_c = Inf)
    ndims(stokes.P) == 2 || return empty
    dt_rsf = typemax(Float64)
    dt_h_min = typemax(Float64)
    dt_w_min = typemax(Float64)
    dt_c_min = typemax(Float64)
    Vp_max = 0.0
    any_cell = false
    if rsf_do_center(fields)
        dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell = _accumulate_gp_dt!(
            dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell,
            Adapt.adapt(Array, fields.Ω),
            Adapt.adapt(Array, fields.a_eff),
            Adapt.adapt(Array, fields.b_eff),
            Adapt.adapt(Array, fields.rsf_mask),
            Adapt.adapt(Array, fields.Vp),
            Adapt.adapt(Array, phase_ratios.center),
            Adapt.adapt(Array, stokes.P),
            rheology,
        )
    end
    if rsf_do_vertex(fields)
        P_c = Adapt.adapt(Array, stokes.P)
        nv = size(fields.rsf_mask_v)
        P_v = zeros(eltype(P_c), nv...)
        nx, ny = size(P_c)
        @inbounds for j in 1:nv[2], i in 1:nv[1]
            i0 = max(i - 1, 1)
            ic = min(i, nx)
            j0 = max(j - 1, 1)
            jc = min(j, ny)
            P_v[i, j] = 0.25 * (P_c[i0, j0] + P_c[ic, jc] + P_c[i0, jc] + P_c[ic, j0])
        end
        dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell = _accumulate_gp_dt!(
            dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell,
            Adapt.adapt(Array, fields.Ωv),
            Adapt.adapt(Array, fields.a_eff_v),
            Adapt.adapt(Array, fields.b_eff_v),
            Adapt.adapt(Array, fields.rsf_mask_v),
            Adapt.adapt(Array, fields.Vp_v),
            Adapt.adapt(Array, phase_ratios.vertex),
            P_v,
            rheology,
        )
    end
    any_cell || return empty
    return (;
        dt_rsf = minimum_mpi(dt_rsf),
        Vp_max = maximum_mpi(Vp_max),
        dt_h = dt_h_min,
        dt_w = dt_w_min,
        dt_c = dt_c_min,
    )
end

function _accumulate_gp_dt!(
        dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell,
        Ω, a_eff, b_eff, mask, Vp_arr, phase_c, P, rheology,
        skip_vs = false,
    )
    ni = size(mask)
    @inbounds for j in 1:ni[2], i in 1:ni[1]
        mask[i, j] ≤ 0.5 && continue
        ip = Int(argmax(phase_c[i, j]))
        comp = rheology[ip].CompositeRheology[1]
        has_rate_state_friction(comp) || continue
        r = rate_state_element(comp)
        Vp_ij = Vp_arr[i, j]
        !(isfinite(Vp_ij) && Vp_ij > 0) && continue
        any_cell = true
        Vp_max = max(Vp_max, Vp_ij)
        Pij = max(abs(P[i, j]), 1.0e3)
        Wf = float(r.D.val)
        L = float(r.L.val)
        V₀ = float(r.V₀.val)
        G = float(shear_modulus_of(comp))
        ν = float(poisson_ratio_of(comp))
        a = a_eff[i, j]
        b = b_eff[i, j]
        dti = compute_dt_ratestate_lamem(;
            Wf = Wf, L = L, V₀ = V₀, a = a, b = b, G = G, ν = ν, P = Pij, Vp = Vp_ij, Ω = Ω[i, j],
        )
        # Velocity-strengthening edges do not set the stable step (LaMEM rsf_dt_skip_vs).
        if !(skip_vs && a >= b)
            dth = dt_healing(; L = L, V₀ = V₀, Ω = Ω[i, j])
            dtw = dt_weakening(;
                L = L, Vp = Vp_ij,
                dteta_max = dteta_max_lamem(a, b, L, Wf, G, ν, Pij),
            )
            dtc = dt_courant(; Wf = Wf, Vp = Vp_ij)
            dt_h_min = min(dt_h_min, dth)
            dt_w_min = min(dt_w_min, dtw)
            dt_c_min = min(dt_c_min, dtc)
            dt_rsf = min(dt_rsf, dti)
        end
    end
    return dt_rsf, dt_h_min, dt_w_min, dt_c_min, Vp_max, any_cell
end

"""
    seas_bp1_tau0(a_max, σ, μ₀, b, V₀, V)

Uniform fault prestress of SEAS BP1-FD: steady-state strength at slip rate `V`
using the largest direct-effect parameter `a_max`.
"""
function seas_bp1_tau0(a_max, σ, μ₀, b, V₀, V)
    arg = (V / (2 * V₀)) * exp((μ₀ + b * log(V₀ / V)) / a_max)
    return σ * a_max * asinh(arg)
end

"""
    seas_bp1_omega(a, b, μ₀, τ, σ, V₀, V)

State `Ω = ln(V₀ θ / L)` such that the local direct effect `a` returns shear
stress `τ` at slip rate `V`. Same inversion as LaMEM `SeasStateAtStress`.
"""
function seas_bp1_omega(a, b, μ₀, τ, σ, V₀, V)
    x = τ / (a * σ)
    x > 0 || throw(ArgumentError("seas_bp1_omega needs τ / (a σ) > 0, got $x"))
    ln_sinh = x > 20 ? x - log(2) : log(sinh(x))
    return (a * (log(2 * V₀ / V) + ln_sinh) - μ₀) / b
end

"""
    init_antiplane_fault!(fields, xvi, xci; y_fault=0, z_rsf_min=-40e3, ...)

Mask the `xy` edges on the fault vertex `y = y_fault` whose cell center is
shallower than `z_rsf_min` (z negative downward). Sets `a(z)`, `b`, and `Ω(z)`
so every masked edge has strength `τ0` at slip rate `V_plate`.

Dirichlet on `Vx`: the outer fault-normal rows, and the two `Vx` nodes that
define the fault jump below `z_rsf_min`, at `±V_plate/2`.
"""
function init_antiplane_fault!(
        fields::JustRelax.RateStateXY,
        xvi,
        xci;
        y_fault = 0.0,
        z_rsf_min = -40.0e3,
        a_knots = (-18.0e3, -15.0e3),
        a_vals = (0.025, 0.010),
        b = 0.015,
        μ₀ = 0.6,
        V₀ = 1.0e-6,
        V_plate = 1.0e-9,
        σ = 50.0e6,
        τ0 = nothing,
    )
    yv = Array(xvi[2])
    zc = Array(xci[3])
    j_fault = argmin(abs.(yv .- y_fault))
    abs(yv[j_fault] - y_fault) < 1.0 ||
        error("no fault-normal vertex within 1 m of y=$y_fault (nearest $(yv[j_fault]))")
    τ0 === nothing && (τ0 = seas_bp1_tau0(maximum(a_vals), σ, μ₀, b, V₀, V_plate))
    isfinite(τ0) && τ0 > 0 || error("BP1 prestress τ0=$τ0 is not a positive finite stress")

    nxy = size(fields.rsf_mask)
    nxy[2] == length(yv) || error("xy edges ($(nxy[2])) do not match vertices ($(length(yv)))")
    nxy[3] == length(zc) || error("xy edges ($(nxy[3])) do not match cell centers ($(length(zc)))")
    Ω = zeros(nxy)
    a_eff = zeros(nxy)
    b_eff = zeros(nxy)
    mask = zeros(nxy)
    @inbounds for k in 1:length(zc), i in 1:nxy[1]
        zc[k] < z_rsf_min && continue
        ak = get_rsf_profile(zc[k], a_knots, a_vals)
        mask[i, j_fault, k] = 1.0
        a_eff[i, j_fault, k] = ak
        b_eff[i, j_fault, k] = b
        Ω[i, j_fault, k] = seas_bp1_omega(ak, b, μ₀, τ0, σ, V₀, V_plate)
    end
    copyto!(fields.Ω, Ω)
    copyto!(fields.Ω_old, Ω)
    copyto!(fields.a_eff, a_eff)
    copyto!(fields.b_eff, b_eff)
    copyto!(fields.rsf_mask, mask)
    fill!(fields.Vp, 0.0)
    fill!(fields.τ_rsf, 0.0)
    fill!(fields.η, 0.0)

    nx_vx, ny_vx, nz_vx = size(fields.vx_fixed)
    ny = length(yv) - 1
    half = 0.5 * V_plate
    fixed = zeros(nx_vx, ny_vx, nz_vx)
    value = zeros(nx_vx, ny_vx, nz_vx)
    @inbounds for k in 1:nz_vx, j in 1:ny_vx, i in 1:nx_vx
        if j <= 2
            fixed[i, j, k] = 1.0
            value[i, j, k] = -half
        elseif j >= ny + 1
            fixed[i, j, k] = 1.0
            value[i, j, k] = half
        end
    end
    @inbounds for k in 1:length(zc)
        zc[k] >= z_rsf_min && continue
        kv = k + 1
        for i in 1:nx_vx
            fixed[i, j_fault, kv] = 1.0
            value[i, j_fault, kv] = -half
            fixed[i, j_fault + 1, kv] = 1.0
            value[i, j_fault + 1, kv] = half
        end
    end
    copyto!(fields.vx_fixed, fixed)
    copyto!(fields.vx_value, value)
    return (; j_fault, τ0, n_rsf = count(>(0.5), mask), half)
end

_gp_rsf_xy_viscosity(::Any) = (nothing, nothing)
_gp_rsf_xy_viscosity(fields::JustRelax.RateStateXY) = (fields.η, fields.rsf_mask)

apply_geoparams_rsf_dirichlet!(::Any, ::Any, ::Any) = nothing
function apply_geoparams_rsf_dirichlet!(stokes, scratch, fields::JustRelax.RateStateXY)
    @parallel (@idx size(fields.vx_fixed)) _apply_vx_dirichlet!(
        stokes.V.Vx, stokes.R.Rx, scratch.dVdτ[1], fields.vx_fixed, fields.vx_value,
    )
    return nothing
end

@parallel_indices (i, j, k) function _apply_vx_dirichlet!(Vx, Rx, dVx, fixed, value)
    @inbounds if fixed[i, j, k] > 0.5
        Vx[i, j, k] = value[i, j, k]
        ir, jr, kr = i - 1, j - 1, k - 1
        if ir ≥ 1 && jr ≥ 1 && kr ≥ 1 &&
                ir ≤ size(Rx, 1) && jr ≤ size(Rx, 2) && kr ≤ size(Rx, 3)
            Rx[ir, jr, kr] = 0.0
            dVx[ir, jr, kr] = 0.0
        end
    end
    return nothing
end

function apply_geoparams_rsf_viscosity!(
        stokes,
        fields::JustRelax.RateStateXY,
        phase_ratios,
        rheology,
        dt;
        viscosity_cutoff = (-Inf, Inf),
        P_floor = 1.0e3,
    )
    ndims(stokes.P) == 3 ||
        error("RateStateXY viscosity is the 3D xy-edge path.")
    η_min, η_max = viscosity_cutoff
    @parallel (@idx size(fields.rsf_mask)) _apply_gp_rsf_xy!(
        fields.η,
        fields.τ_rsf,
        stokes.ε.xy,
        stokes.τ_o.xy,
        stokes.P,
        fields.Ω_old,
        fields.a_eff,
        fields.b_eff,
        fields.rsf_mask,
        phase_ratios.xy,
        rheology,
        dt,
        P_floor,
        η_min,
        η_max,
        periodic_dims(stokes),
    )
    return nothing
end

@parallel_indices (i, j, k) function _apply_gp_rsf_xy!(
        η, τ_rsf, εxy, τxy, P,
        Ω_old, a_eff, b_eff, rsf_mask, phase_xy, rheology,
        dt, P_floor, η_min, η_max, periodic,
    )
    @inbounds if rsf_mask[i, j, k] > 0.5
        ip = _dominant_phase(phase_xy, i, j, k)
        ni = size(P)
        Ic = clamped_indices(ni, periodic, i, j, k)
        # Fault traction is the xy shear. The momentum residual reads τ.xy from
        # this edge; the other tensor components at the edge are not stored.
        z = zero(εxy[i, j, k])
        active, η_new, τ_new = _gp_rsf_maxwell(
            rheology, ip,
            z, z, εxy[i, j, k],
            z, z, τxy[i, j, k],
            max(abs(av_clamped_xy(P, Ic...)), P_floor),
            Ω_old[i, j, k], a_eff[i, j, k], b_eff[i, j, k], dt,
        )
        if active && isfinite(η_new) && η_new > 0
            η[i, j, k] = min(max(η_new, η_min), η_max)
            τ_rsf[i, j, k] = τ_new
        end
    end
    return nothing
end

function update_geoparams_rate_state!(
        fields::JustRelax.RateStateXY, stokes, phase_ratios, rheology, dt;
        P_floor = 1.0e3,
    )
    ndims(stokes.P) == 3 ||
        error("RateStateXY state update is the 3D xy-edge path.")
    @parallel (@idx size(fields.rsf_mask)) _update_gp_rsf_xy!(
        fields.Ω, fields.Ω_old, fields.Vp,
        fields.a_eff, fields.b_eff, fields.rsf_mask,
        stokes.τ.xy, stokes.P,
        phase_ratios.xy, rheology, dt, P_floor, periodic_dims(stokes),
    )
    return nothing
end

@parallel_indices (i, j, k) function _update_gp_rsf_xy!(
        Ω, Ω_old, Vp, a_eff, b_eff, rsf_mask, τxy, P,
        phase_xy, rheology, dt, P_floor, periodic,
    )
    @inbounds if rsf_mask[i, j, k] > 0.5
        ip = _dominant_phase(phase_xy, i, j, k)
        ni = size(P)
        Ic = clamped_indices(ni, periodic, i, j, k)
        active, Vp_ijk, Ωnew = _gp_rsf_advance(
            rheology, ip, abs(τxy[i, j, k]),
            max(abs(av_clamped_xy(P, Ic...)), P_floor),
            Ω_old[i, j, k], a_eff[i, j, k], b_eff[i, j, k], dt,
        )
        if active && isfinite(Ωnew)
            Ω[i, j, k] = Ωnew
            Ω_old[i, j, k] = Ωnew
            Vp[i, j, k] = Vp_ijk
        end
    end
    return nothing
end

function compute_dt_geoparams_rsf(
        fields::JustRelax.RateStateXY, stokes, phase_ratios, rheology, dt,
    )
    empty = (; dt_rsf = Inf, Vp_max = 0.0, dt_h = Inf, dt_w = Inf, dt_c = Inf)
    mask = Adapt.adapt(Array, fields.rsf_mask)
    Ω = Adapt.adapt(Array, fields.Ω)
    a_eff = Adapt.adapt(Array, fields.a_eff)
    b_eff = Adapt.adapt(Array, fields.b_eff)
    Vp_arr = Adapt.adapt(Array, fields.Vp)
    phase_xy = Adapt.adapt(Array, phase_ratios.xy)
    P = Adapt.adapt(Array, stokes.P)
    periodic = periodic_dims(stokes)
    ni = size(P)
    dt_rsf = typemax(Float64)
    dt_h_min = typemax(Float64)
    dt_w_min = typemax(Float64)
    dt_c_min = typemax(Float64)
    Vp_max = 0.0
    any_cell = false
    @inbounds for k in 1:size(mask, 3), j in 1:size(mask, 2), i in 1:size(mask, 1)
        mask[i, j, k] ≤ 0.5 && continue
        ip = _dominant_phase(phase_xy, i, j, k)
        comp = rheology[ip].CompositeRheology[1]
        has_rate_state_friction(comp) || continue
        Vp_ij = Vp_arr[i, j, k]
        !(isfinite(Vp_ij) && Vp_ij > 0) && continue
        any_cell = true
        Vp_max = max(Vp_max, Vp_ij)
        Ic = clamped_indices(ni, periodic, i, j, k)
        Pij = max(abs(av_clamped_xy(P, Ic...)), 1.0e3)
        r = rate_state_element(comp)
        a = a_eff[i, j, k]
        b = b_eff[i, j, k]
        fields.skip_vs && a >= b && continue
        Wf = float(r.D.val)
        L = float(r.L.val)
        V₀ = float(r.V₀.val)
        G = float(shear_modulus_of(comp))
        ν = float(poisson_ratio_of(comp))
        Ωij = Ω[i, j, k]
        dti = compute_dt_ratestate_lamem(;
            Wf = Wf, L = L, V₀ = V₀, a = a, b = b, G = G, ν = ν, P = Pij, Vp = Vp_ij, Ω = Ωij,
        )
        dt_h_min = min(dt_h_min, dt_healing(; L = L, V₀ = V₀, Ω = Ωij))
        dt_w_min = min(
            dt_w_min,
            dt_weakening(;
                L = L, Vp = Vp_ij,
                dteta_max = dteta_max_lamem(a, b, L, Wf, G, ν, Pij),
            ),
        )
        dt_c_min = min(dt_c_min, dt_courant(; Wf = Wf, Vp = Vp_ij))
        dt_rsf = min(dt_rsf, dti)
    end
    any_cell || return empty
    return (;
        dt_rsf = minimum_mpi(dt_rsf),
        Vp_max = maximum_mpi(Vp_max),
        dt_h = dt_h_min,
        dt_w = dt_w_min,
        dt_c = dt_c_min,
    )
end
