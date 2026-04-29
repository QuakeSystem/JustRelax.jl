function compute_stress_DRYEL!(
        stokes,
        rheology,
        phase_ratios,
        λ_relaxation,
        dt;
        rsf_params = nothing,
        rsf_state = nothing,
        periodic_x::Bool = false,
    )
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy_c), # centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        rheology,
        phase_ratios.center,
        phase_ratios.vertex,
        λ_relaxation,
        dt,
        rsf_params,
        rsf_state,
        periodic_x,
    )
    return nothing
end

Base.@propagate_inbounds @inline function periodic_indices_x(ni::NTuple{2, Integer}, i, j)
    nx, ny = ni
    i0 = mod1(i - 1, nx)
    ic = mod1(i, nx)
    j0 = clamp(j - 1, 1, ny)
    jc = clamp(j, 1, ny)
    return i0, j0, ic, jc
end

@inline _rsf_pick(v::Number, ::Int) = v
@inline _rsf_pick(v::NTuple{N, <:Number}, phase::Int) where {N} = v[phase]
@inline _rsf_pick(v::AbstractVector{<:Number}, phase::Int) = v[phase]
@inline _rsf_phase_on(rsf_params, phase::Int) =
    hasproperty(rsf_params, :active) ? Bool(_rsf_pick(rsf_params.active, phase)) : true

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        P,
        λ,
        λv,
        η,
        ηv,
        η_vep,
        ΔPψ,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt, rsf_params, rsf_state, periodic_x
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    ## VERTEX CALCULATION
    @inbounds begin
        Ic = periodic_x ? periodic_indices_x(ni, I...) : clamped_indices(ni, I...)
        τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
        εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
        λvij = λv[I...]
        # ηij   = harm_clamped(η, Ic...)
        ηij = ηv[I...]
        Pij = av_clamped(P, Ic...)
        ratio = phase_ratios_vertex[I...]
        # compute local stress
        Ω_old_v = rsf_state === nothing ? 0.0 : rsf_state.Ω_vertex[I...]
        τxx_I, τyy_I, τxy_I, _, _, _, _, λ_I, _, _, Ω_new_v, Vp_v = compute_local_stress(
            εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, rsf_params, Ω_old_v
        )

        # update arrays
        τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
        λv[I...] = λ_I
        if rsf_state !== nothing
            rsf_state.Ω_vertex_new[I...] = Ω_new_v
            rsf_state.Vp_vertex[I...] = Vp_v
        end

        ## CENTER CALCULATION
        if all(I .≤ ni)
            τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
            εij = ε[1][I...], ε[2][I...], av(ε[3])
            λij = λ[I...]
            ηij = η[I...]
            Pij = P[I...]
            ratio = phase_ratios_center[I...]

            # compute local stress
            Ω_old_c = rsf_state === nothing ? 0.0 : rsf_state.Ω_center[I...]
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, Ω_new_c, Vp_c = compute_local_stress(
                εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, rsf_params, Ω_old_c
            )
            # update arrays
            τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = εxx_pl, εyy_pl, εxy_pl
            τII[I...] = τII_I
            η_vep[I...] = ηvep_I
            λ[I...] = λ_I
            ΔPψ[I...] = ΔPψ_I
            if rsf_state !== nothing
                rsf_state.Ω_center_new[I...] = Ω_new_c
                rsf_state.Vp_center[I...] = Vp_c
            end
        end
    end

    return nothing
end

@generated function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio::SVector{N}, dt, rsf_params, Ω_old) where {N}
    return quote
        @inline
        # iterate over phases
        v_phases = Base.@ntuple $N phase -> begin
            # get phase ratio
            ratio_I = phase_ratio[phase]
            v = if iszero(ratio_I) # this phase does not contribute
                empty_stress_solution(εij)

            else
                # get rheological properties for this phase
                G = get_shear_modulus(rheology, phase)
                Kb = get_bulk_modulus(rheology, phase)
                ispl, C, sinϕ, cosϕ, sinΨ, η_reg = plastic_params(rheology[phase].CompositeRheology[1].elements, 0.0e0) # this 0e0 is accumulated plastic strain, not used here
                # compute local stress
                ratio_I .* _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, dt, rsf_params, phase, Ω_old)
            end
        end
        # sum contributions from all phases
        v = reduce(.+, v_phases)
        return v # this returns (τ_ij...), (εij_pl...), τII, λ, ΔPψ, ηvep
    end
end

@inline function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio, dt, rsf_params)
    return compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio, dt, rsf_params, 0.0)
end

@inline function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio, dt)
    return compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio, dt, nothing, 0.0)
end

@inline function _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, dt, rsf_params, phase::Int, Ω_old)

    # viscoelastic viscosity
    η_ve = isinf(G) ?
        inv(inv(η) + inv(G * dt)) :
        (η * G * dt) / (η + G * dt) # more efficient than inv(inv(η) + inv(G * dt))
    # effective strain rate
    inv_2Gdt = inv(2 * G * dt)
    εij_eff = @. εij + τij_o * inv_2Gdt

    εII = second_invariant(εij_eff)

    # early return if there is no deformation
    iszero(εII) && return (zero_tuple(εij)..., zero_tuple(εij)..., 0.0, 0.0, 0.0, η, Ω_old, 0.0)

    # Plastic stress correction starts here
    τij = @. 2 * η_ve * εij_eff
    τII = second_invariant(τij)
    if !(isfinite(τII) && isfinite(η_ve) && isfinite(εII))
        return (zero_tuple(εij)..., zero_tuple(εij)..., 0.0, 0.0, 0.0, η, Ω_old, 0.0)
    end
    rsf_on = (!ispl) && rsf_params !== nothing && _rsf_phase_on(rsf_params, phase)
    p_shift = rsf_on && hasproperty(rsf_params, :p_shift) ? _rsf_pick(rsf_params.p_shift, phase) : 0.0
    P_eff = P + p_shift
    if rsf_on && P_eff > 0.0
        if hasproperty(rsf_params, :a) && hasproperty(rsf_params, :b) && hasproperty(rsf_params, :μ0)
            a = _rsf_pick(rsf_params.a, phase)
            b = _rsf_pick(rsf_params.b, phase)
            μ0 = _rsf_pick(rsf_params.μ0, phase)
            L = _rsf_pick(rsf_params.L, phase)
            D = _rsf_pick(rsf_params.D, phase)
            λ_rsf = hasproperty(rsf_params, :λ) ? _rsf_pick(rsf_params.λ, phase) : 0.0
            C_rsf = hasproperty(rsf_params, :C) ? _rsf_pick(rsf_params.C, phase) : 0.0
            V0 = hasproperty(rsf_params, :V0) ? _rsf_pick(rsf_params.V0, phase) :
                (hasproperty(rsf_params, :V0_model) ? rsf_params.V0_model : 1.0e-9)
            η_min = hasproperty(rsf_params, :η_min) ? _rsf_pick(rsf_params.η_min, phase) : 1.0e18
            η_max = hasproperty(rsf_params, :η_max) ? _rsf_pick(rsf_params.η_max, phase) : 1.0e23
            P_rsf = max(P * (1.0 - λ_rsf) + p_shift, eps(Float64))

            # RSF viscous form driven by local strain rate (stable in Stokes solve).
            Vp = 2.0 * D * εII
            # println("Vp = $Vp")
            Vp_max = hasproperty(rsf_params, :Vp_max) ? _rsf_pick(rsf_params.Vp_max, phase) : 1.0e19
            Vp = clamp(Vp, 0.0, Vp_max)

            exp_arg = (μ0 + b * Ω_old) / max(a, eps(Float64))
            # exp_arg = min(exp_arg, 700.0)
            μd = a * asinh(Vp / max(2.0 * V0, eps(Float64)) * exp(exp_arg))
            τII_rsf = P_rsf * μd + C_rsf
            η_rsf = τII_rsf / max(2.0 * εII, eps(Float64))
            η_rsf = clamp(η_rsf, η_min, η_max)
            η_vep = min(η_ve, η_rsf)
            if !isfinite(η_vep)
                η_vep = η
            end
            τij = @. 2 * η_vep * εij_eff
            τII = second_invariant(τij)
            if !isfinite(τII)
                return (zero_tuple(εij)..., zero_tuple(εij)..., 0.0, 0.0, 0.0, η, Ω_old, 0.0)
            end
            εij_pl = zero_tuple(εij)
            Vp = 2.0 * V0 * sinh(max(τII - C_rsf, 0.0) / (a * P_rsf)) * exp(-exp_arg)
            var_rsf = (Vp * dt) / max(L, eps(Float64))
            Ω = if var_rsf <= 1.0e-6
                log(exp(Ω_old) * (1.0 - var_rsf) + V0 * dt / max(L, eps(Float64)))
            else
                log(V0 / max(Vp, eps(Float64)) + (exp(Ω_old) - V0 / max(Vp, eps(Float64))) * exp(-var_rsf))
            end
            Ω = clamp(Ω, -100.0, 100.0)
            # println("Ω = $Ω")
            # println("P_rsf = $P_rsf")
            # println("Vp = $Vp")
            # println("dt = $dt")
            # println("var_rsf = $var_rsf")
            # println("μd = $μd")
            # println("τII = $τII")
            # error("stop")   
            if !isfinite(Ω)
                Ω = Ω_old
            end
            λ = 0.0
            ΔPψ = 0.0
            return τij..., εij_pl..., τII, λ, ΔPψ, η_vep, Ω, Vp
        end

        μs = _rsf_pick(rsf_params.mu_s, phase)
        μd = clamp(_rsf_pick(rsf_params.mu_d, phase), 0.0, μs)
        σc = _rsf_pick(rsf_params.sigma_c, phase)
        Vc = _rsf_pick(rsf_params.Vc, phase)
        D = _rsf_pick(rsf_params.D, phase)
        maxit = hasproperty(rsf_params, :maxit) ? Int(_rsf_pick(rsf_params.maxit, phase)) : 50
        rtol = hasproperty(rsf_params, :rtol) ? _rsf_pick(rsf_params.rtol, phase) : 1.0e-5
        τII_trial = τII
        τy = P_eff * (μd <= μs ? μd : μs) + σc
        η_test = τy / (2 * εII)
        DIIpl = max(εII - (η_test / max(η_ve, eps())) * εII, 0.0)
        for _ in 1:maxit
            Vp = 2.0 * D * DIIpl
            μeff = μd + (μs - μd) / (1.0 + Vp / Vc)
            τy = P_eff * μeff + σc
            η_test = τy / (2 * εII)
            DIIpl_new = max(εII - (η_test / max(η_ve, eps())) * εII, 0.0)
            abs(DIIpl_new - DIIpl) <= rtol * (εII + eps()) && (DIIpl = DIIpl_new; break)
            DIIpl = DIIpl_new
        end
        η_vep = min(η_ve, τy / (2 * εII))
        τij = @. 2 * η_vep * εij_eff
        τII = second_invariant(τij)
        εij_pl = if DIIpl > 0.0 && τII > 0.0
            @. DIIpl * 0.5 * τij / τII
        else
            zero_tuple(εij)
        end
        λ = 0.0
        ΔPψ = 0.0
        return τij..., εij_pl..., τII, λ, ΔPψ, η_vep, Ω_old, Vp
    end

    # Drucker-Prager regularized update.
    F = τII - C * cosϕ - P * sinϕ
    λ = if ispl && F ≥ 0
        λ_new = F / (η_ve + η_reg + Kb * dt * sinϕ * sinΨ)
        λ_relaxation * λ_new + (1 - λ_relaxation) * λ
    else
        0.0
    end
    # Effective viscoelastic-plastic viscosity
    η_vep = (τII - λ * η_ve) / (2 * εII)
    if !isfinite(η_vep)
        η_vep = η
    end
    # Update stress and plastic strain rate
    τij, τII, εij_pl, ΔPψ = if λ > 0
        τij = @. 2 * η_vep * εij_eff
        τII = second_invariant(τij)
        if !isfinite(τII) || iszero(τII)
            τij = zero_tuple(εij)
            τII = 0.0
            εij_pl = zero_tuple(εij)
            ΔPψ = 0.0
            τij, τII, εij_pl, ΔPψ
        else
        εij_pl = @. λ * 0.5 * τij / τII # λ * dQdτij
        # Update pressure correction due to dilatation
        ΔPψ = iszero(sinΨ) ? 0.0 : λ * sinΨ * Kb * dt
        τij, τII, εij_pl, ΔPψ
        end
    else
        εij_pl = zero_tuple(εij)
        ΔPψ = 0.0
        τij, τII, εij_pl, ΔPψ
    end

    return τij..., εij_pl..., τII, λ, ΔPψ, η_vep, Ω_old, 0.0
end

# this returns zero for: τxx, τyy, τxy, εxx_pl, εyy_pl, εxy_pl, τII, λ, ΔPψ, ηvep, Ω, Vp
@inline empty_stress_solution(::NTuple{3, T}) where {T} = zero_tuple(T, Val(12))
# this returns zero for: τxx, τyy, τzz, τyz, τxz, τxy, εpl..., τII, λ, ΔPψ, ηvep, Ω, Vp
@inline empty_stress_solution(::NTuple{6, T}) where {T} = zero_tuple(T, Val(15))

@inline zero_tuple(::Type{T}, ::Val{N}) where {T, N} = ntuple(_ -> zero(T), Val(N))
@inline zero_tuple(::NTuple{N, T}) where {T, N} = zero_tuple(T, Val(N))


## VARIATIONAL STOKES STRESS KERNELS

function compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy_c), # centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt
    )
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        P,
        λ,
        λv,
        η,
        η_vep,
        ΔPψ,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    @inbounds begin
        ## VERTEX CALCULATION
        @inbounds if isvalid_v(ϕ, I...)
            Ic = clamped_indices(ni, I...)
            τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
            εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
            λvij = λv[I...]
            ηij = harm_clamped(η, Ic...)
            Pij = av_clamped(P, Ic...)
            ratio = phase_ratios_vertex[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, _, _, _, _, λ_I, _, _, _, _ = compute_local_stress(
                εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt
            )

            # update arrays
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
            λv[I...] = λ_I

        else
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = 0.0e0, 0.0e0, 0.0e0
            λv[I...] = 0.0e0

        end

        ## CENTER CALCULATION
        if all(I .≤ ni)
            @inbounds if isvalid_c(ϕ, I...)
                τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
                εij = ε[1][I...], ε[2][I...], av(ε[3])
                λij = λ[I...]
                ηij = η[I...]
                Pij = P[I...]
                ratio = phase_ratios_center[I...]

                # compute local stress
                τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, _, _ = compute_local_stress(
                    εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt
                )
                # update arrays
                τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
                ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = εxx_pl, εyy_pl, εxy_pl
                τII[I...] = τII_I
                η_vep[I...] = ηvep_I
                λ[I...] = λ_I
                ΔPψ[I...] = ΔPψ_I

            else
                τ[1][I...], τ[2][I...], τ[3][I...] = 0.0e0, 0.0e0, 0.0e0
                ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = 0.0e0, 0.0e0, 0.0e0
                τII[I...] = 0.0e0
                η_vep[I...] = 0.0e0
                λ[I...] = 0.0e0
                ΔPψ[I...] = 0.0e0

            end
        end
    end

    return nothing
end
