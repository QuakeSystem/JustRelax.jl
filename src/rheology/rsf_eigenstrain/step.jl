# One physical step of the eigenstrain split: elastic DYREL with frozen V_lag,
# then a slip-rate update, at most `max_pass` times. State is aged once at the end.
# Inertia uses the velocity from the previous accepted step, not the previous pass.

const _EIGEN_τ_FIELDS = (:xx, :yy, :zz, :xx_v, :yy_v, :zz_v, :xy, :yz, :xz, :xy_c, :yz_c, :xz_c)

function _copy_stress_fields!(dst, src)
    for name in _EIGEN_τ_FIELDS
        a = getfield(src, name)
        a === nothing && continue
        copyto!(getfield(dst, name), a)
    end
    return nothing
end

function _similar_stress(τ)
    c(a) = a === nothing ? nothing : similar(a)
    return JustRelax.SymmetricTensor(
        c(τ.xx), c(τ.yy), c(τ.zz),
        c(τ.xx_v), c(τ.yy_v), c(τ.zz_v),
        c(τ.xy), c(τ.yz), c(τ.xz),
        c(τ.xy_c), c(τ.yz_c), c(τ.xz_c),
        similar(τ.II),
    )
end

function _similar_velocity(V)
    return JustRelax.Velocity(
        similar(V.Vx), similar(V.Vy), V.Vz === nothing ? nothing : similar(V.Vz),
    )
end

function _copy_velocity!(dst, src)
    copyto!(dst.Vx, src.Vx)
    copyto!(dst.Vy, src.Vy)
    src.Vz === nothing || copyto!(dst.Vz, src.Vz)
    return nothing
end

"""
    eigenstrain_physical_step!(stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg, ez;
        inertia=false, ρ_inertia=nothing, solve_kwargs=(;))

Relax the elastic problem with `ez.V_lag` and `ez.ψ_old` frozen, then invert the
slip rate that is consistent with unloading `τ` by `(G Δt / Wf)(V − V_lag)`.
Repeat until the fault slip rate changes by less than `ez.tol` or `ez.max_pass`
is reached. Then age `ψ` once and commit `ψ_old` and `V_lag`. The inversion is
the shared kernel, so a CUDA or AMDGPU array takes the same bracket.

`stokes.τ_o` and, when `inertia` is set, `stokes.V0` are restored at the start of
every pass so the history stays that of the last accepted physical step.
"""
function eigenstrain_physical_step!(
        stokes::JustRelax.StokesArrays, args...; kwargs...,
    )
    return eigenstrain_physical_step!(backend(stokes), stokes, args...; kwargs...)
end

eigenstrain_physical_step!(::CPUBackendTrait, stokes, args...; kwargs...) =
    _eigenstrain_physical_step!(stokes, args...; kwargs...)

function _eigenstrain_physical_step!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg, ez::JustRelax.EigenstrainRSF;
        inertia = false,
        ρ_inertia = nothing,
        solve_kwargs = (;),
    )
    τ_snap = _similar_stress(stokes.τ_o)
    _copy_stress_fields!(τ_snap, stokes.τ_o)
    V0_snap = _similar_velocity(stokes.V0)
    _copy_velocity!(V0_snap, stokes.V0)

    rel = Inf
    passes = 0
    for pass in 1:ez.max_pass
        passes = pass
        _copy_stress_fields!(stokes.τ_o, τ_snap)
        if inertia
            _copy_velocity!(stokes.V0, V0_snap)
        end
        # Local solver: this function already runs in the module selected by `backend(stokes)`.
        _solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            flatten_solver_kwargs(solve_kwargs)...,
            rsf_eigenstrain = ez,
            inertia = inertia,
            ρ_inertia = ρ_inertia,
        )
        invert_rsf_eigenstrain!(ez, stokes, dt)
        rel = eigenstrain_slip_change(ez)
        rel < ez.tol && break
        copyto!(ez.V_lag, ez.V)
    end
    age_rsf_eigenstrain!(ez, dt)
    copyto!(ez.ψ_old, ez.ψ)
    copyto!(ez.V_lag, ez.V)
    dt_rsf = eigenstrain_dt_limit(ez)
    V_max = isfinite(dt_rsf) ? 0.2 * ez.law.L / dt_rsf : 0.0
    return (; passes = passes, rel = rel, dt_rsf = dt_rsf, V_max = V_max)
end
