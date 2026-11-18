# Vertex kernels for the eigenstrain fault row. Included from every dimension module
# (`common.jl`); the index tuple `I...` follows the array rank.

"""
    eigenstrain_rsf(ni; Wf, G, σ, law, V_cap=1000, tol=1e-2, max_pass=4)
    eigenstrain_rsf(backend, ni; ...)

Allocate a vertex-sized [`EigenstrainRSF`](@ref) with this module's `@zeros`.
`eigenstrain_rsf(backend, ni)` allocates on that backend. CUDA and AMDGPU
extensions override the `CUDABackend` and `AMDGPUBackend` methods.
"""
function eigenstrain_rsf(
        ni::NTuple{N, Integer};
        Wf,
        G,
        σ,
        law::JustRelax.RateStateFriction,
        V_cap = 1000.0,
        tol = 1.0e-2,
        max_pass = 4,
    ) where {N}
    Wf > 0 || throw(ArgumentError("Wf must be positive, got $Wf"))
    max_pass ≥ 1 || throw(ArgumentError("max_pass must be ≥ 1, got $max_pass"))
    nv = ni .+ 1
    mask = @zeros(nv...)
    ψ = @zeros(nv...)
    ψ_old = @zeros(nv...)
    V_lag = @zeros(nv...)
    V = @zeros(nv...)
    a = @zeros(nv...)
    b = @zeros(nv...)
    T = eltype(mask)
    return JustRelax.EigenstrainRSF(
        mask, ψ, ψ_old, V_lag, V, a, b,
        T(Wf), T(G), T(σ), T(V_cap), T(tol), Int(max_pass), law,
    )
end

eigenstrain_rsf(backend, ni::NTuple{N, Integer}; kwargs...) where {N} =
    eigenstrain_rsf(ni; kwargs...)

"""
    init_eigenstrain_row!(ez, xvi; y_fault, a, b, ψ, V_lag,
        a_x=nothing, a_val=nothing, b_x=nothing, b_val=nothing, x_creep=nothing)

Mark the vertex row closest to `y_fault` and set `a`, `b`, `ψ` / `ψ_old`, and `V_lag`.
`V` on that row starts equal to `V_lag`.

`a_x` / `a_val` and `b_x` / `b_val` are optional knots (strictly increasing). Between
and outside them, the value follows [`get_rsf_profile`](@ref). Without knots, `a` and
`b` are the scalars. Nodes with `x <= x_creep` are prescribed plate-rate nodes
(`mask = 2`); the rest of the row is rate-and-state (`mask = 1`).
"""
function init_eigenstrain_row!(
        ez::JustRelax.EigenstrainRSF, xvi;
        y_fault, a, b, ψ, V_lag,
        a_x = nothing, a_val = nothing,
        b_x = nothing, b_val = nothing,
        x_creep = nothing,
    )
    x = Array(xvi[1])
    y = Array(xvi[2])
    j_fault = argmin(abs.(y .- y_fault))
    mask = zeros(eltype(ez.mask), size(ez.mask))
    aa = zeros(eltype(ez.a), size(ez.a))
    bb = zeros(eltype(ez.b), size(ez.b))
    ψa = zeros(eltype(ez.ψ), size(ez.ψ))
    Vl = zeros(eltype(ez.V_lag), size(ez.V_lag))
    use_a = !isnothing(a_x) && !isnothing(a_val)
    use_b = !isnothing(b_x) && !isnothing(b_val)
    @inbounds for i in axes(mask, 1)
        ai = use_a ? get_rsf_profile(x[i], a_x, a_val) : a
        bi = use_b ? get_rsf_profile(x[i], b_x, b_val) : b
        creep = !isnothing(x_creep) && x[i] ≤ x_creep
        mask[i, j_fault] = creep ? 2 : 1
        aa[i, j_fault] = ai
        bb[i, j_fault] = bi
        ψa[i, j_fault] = ψ
        Vl[i, j_fault] = V_lag
    end
    copyto!(ez.mask, mask)
    copyto!(ez.a, aa)
    copyto!(ez.b, bb)
    copyto!(ez.ψ, ψa)
    copyto!(ez.ψ_old, ψa)
    copyto!(ez.V_lag, Vl)
    copyto!(ez.V, Vl)
    return j_fault
end

@parallel_indices (I...) function _apply_eigenstrain_τ!(τxy, εxy, τxy_o, mask, V_lag, G, dt, Wf)
    if mask[I...] > 0.5
        τ_kernel = τxy[I...]
        τ_new = eigenstrain_trial_stress(τxy_o[I...], εxy[I...], V_lag[I...], G, dt, Wf)
        # Stay on the kernel traction's side of zero, and do not exceed it.
        τxy[I...] = ifelse(
            τ_kernel > 0,
            min(τ_kernel, max(τ_new, zero(τ_kernel))),
            ifelse(
                τ_kernel < 0,
                max(τ_kernel, min(τ_new, zero(τ_kernel))),
                zero(τ_kernel),
            ),
        )
    end
    return nothing
end

"""Device implementation. The solver calls this directly so it stays on the module that owns the kernels."""
function _apply_rsf_eigenstrain!(stokes, ez, dt)
    ez === nothing && return nothing
    T = eltype(stokes.τ.xy)
    @parallel (@idx size(stokes.τ.xy)) _apply_eigenstrain_τ!(
        stokes.τ.xy, stokes.ε.xy, stokes.τ_o.xy,
        ez.mask, ez.V_lag,
        T(ez.G), T(dt), T(ez.Wf),
    )
    return nothing
end

"""Overwrite fault `τ.xy` with the frozen-`V_lag` eigenstrain traction. No-op if `ez` is `nothing`."""
function apply_rsf_eigenstrain!(stokes, ez, dt)
    ez === nothing && return nothing
    return apply_rsf_eigenstrain!(backend(stokes), stokes, ez, dt)
end

apply_rsf_eigenstrain!(::CPUBackendTrait, stokes, ez, dt) =
    _apply_rsf_eigenstrain!(stokes, ez, dt)

@parallel_indices (I...) function _invert_eigenstrain_V!(V, τxy, mask, ψ_old, a, b, V_lag, proto, σ, V_cap)
    if mask[I...] > 1.5
        # Prescribed slip (BP1-FD below Wf_height): keep the imposed rate.
        V[I...] = V_lag[I...]
    elseif mask[I...] > 0.5
        V[I...] = eigenstrain_slip_rate(proto, a[I...], b[I...], τxy[I...], ψ_old[I...], σ, V_cap)
    else
        V[I...] = 0
    end
    return nothing
end

"""Invert `|τ.xy|` on rate-and-state fault nodes into `ez.V`. Prescribed nodes keep `V_lag`."""
function invert_rsf_eigenstrain!(ez::JustRelax.EigenstrainRSF, stokes)
    T = eltype(ez.V)
    @parallel (@idx size(ez.V)) _invert_eigenstrain_V!(
        ez.V, stokes.τ.xy, ez.mask, ez.ψ_old, ez.a, ez.b, ez.V_lag,
        ez.law, T(ez.σ), T(ez.V_cap),
    )
    return nothing
end

@parallel_indices (I...) function _age_eigenstrain_ψ!(ψ, ψ_old, V, mask, a, b, proto, dt)
    if mask[I...] > 0.5 && mask[I...] < 1.5
        ψ[I...] = eigenstrain_age_state(proto, a[I...], b[I...], ψ_old[I...], V[I...], dt)
    else
        ψ[I...] = ψ_old[I...]
    end
    return nothing
end

"""Age `ψ` once from `ψ_old` and the current slip rate `V`. Does not commit `ψ_old`."""
function age_rsf_eigenstrain!(ez::JustRelax.EigenstrainRSF, dt)
    T = eltype(ez.ψ)
    @parallel (@idx size(ez.ψ)) _age_eigenstrain_ψ!(
        ez.ψ, ez.ψ_old, ez.V, ez.mask, ez.a, ez.b, ez.law, T(dt),
    )
    return nothing
end

"""
    set_eigenstrain_ψ_from_stress!(ez, τ, V)

On the fault row, set `ψ` and `ψ_old` so the slip law returns `V` at traction `τ`.
"""
function set_eigenstrain_ψ_from_stress!(ez::JustRelax.EigenstrainRSF, τ, V)
    ψ = Array(ez.ψ)
    a = Array(ez.a)
    b = Array(ez.b)
    mask = Array(ez.mask)
    law = ez.law
    @inbounds for i in eachindex(ψ, a, b, mask)
        mask[i] > 0.5 && mask[i] < 1.5 || continue
        ψ[i] = eigenstrain_ψ_for_slip(a[i], b[i], law.μ₀, law.V₀, V, τ, ez.σ)
    end
    copyto!(ez.ψ, ψ)
    copyto!(ez.ψ_old, ψ)
    return nothing
end

"""Maximum `|V − V_lag| / max(V, V_lag)` on the fault row."""
function eigenstrain_slip_change(ez::JustRelax.EigenstrainRSF)
    V = Array(ez.V)
    V_lag = Array(ez.V_lag)
    mask = Array(ez.mask)
    rel = zero(eltype(V))
    @inbounds for i in eachindex(V, V_lag, mask)
        mask[i] > 0.5 || continue
        denom = max(V[i], V_lag[i], eps(eltype(V)))
        rel = max(rel, abs(V[i] - V_lag[i]) / denom)
    end
    return rel
end

"""
Smallest LaMEM rate-and-state timestep on the fault row.

`min(dt_weakening, dt_healing, dt_slip)` on rate-and-state nodes (`mask = 1`).
Prescribed plate-rate nodes are omitted. While the fault is locked, healing
(`0.2 L / (V₀ exp(-ψ))`) sets the step. During fast slip, weakening and the
slip Courant number take over.
"""
function eigenstrain_lamem_dt(ez::JustRelax.EigenstrainRSF; ν)
    V = Array(ez.V)
    ψ = Array(ez.ψ)
    a = Array(ez.a)
    b = Array(ez.b)
    mask = Array(ez.mask)
    law = ez.law
    dt = Inf
    @inbounds for i in eachindex(V, ψ, a, b, mask)
        mask[i] > 0.5 && mask[i] < 1.5 || continue
        dti = compute_dt_ratestate_lamem(;
            Wf = ez.Wf,
            L = law.L,
            V₀ = law.V₀,
            a = a[i],
            b = b[i],
            G = ez.G,
            ν = ν,
            P = ez.σ,
            Vp = V[i],
            Ω = ψ[i],
        )
        dt = min(dt, dti)
    end
    return dt
end

"""`0.2 L / V_max` on the fault row, or `Inf` when the fault is stuck at `V = 0`."""
function eigenstrain_dt_limit(ez::JustRelax.EigenstrainRSF)
    V = Array(ez.V)
    mask = Array(ez.mask)
    V_max = zero(eltype(V))
    @inbounds for i in eachindex(V, mask)
        mask[i] > 0.5 && (V_max = max(V_max, V[i]))
    end
    return V_max > 0 ? 0.2 * ez.law.L / V_max : Inf
end
