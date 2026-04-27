# # Traits

const PERIODIC_X_VISCOSITY_ARGS = Ref(false)

@inline function set_periodic_x_viscosity_args!(flag::Bool)
    PERIODIC_X_VISCOSITY_ARGS[] = flag
    return nothing
end

# without phase ratios
@inline function update_viscosity_εII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        stokes, args, rheology, cutoff, compute_viscosity_εII; relaxation = relaxation
    )
    return nothing
end

@inline function update_viscosity_τII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    update_viscosity!(
        stokes, args, rheology, cutoff, compute_viscosity_τII; relaxation = relaxation
    )
    return nothing
end

# @inline update_viscosity!(::LinearRheologyTrait, args::Vararg{Any, N}; relaxation = 1.0e0) where {N} = nothing

@inline function update_viscosity!(
        stokes::JustRelax.StokesArrays,
        args,
        rheology,
        cutoff,
        fn_viscosity::F;
        relaxation = 1.0e0,
    ) where {F}

    fn = get_viscosity_fn(fn_viscosity)

    fn(stokes, args, rheology, cutoff, fn_viscosity; relaxation = relaxation)

    return nothing
end

@inline get_viscosity_fn(::typeof(compute_viscosity_εII)) = compute_viscosity_εII!
@inline get_viscosity_fn(::typeof(compute_viscosity_τII)) = compute_viscosity_τII!

# with phase ratios

@inline function update_viscosity_εII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    update_viscosity!(
        stokes,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff,
        compute_viscosity_εII;
        relaxation = relaxation,
    )
    return nothing
end

@inline function update_viscosity_τII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    update_viscosity!(
        stokes,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff,
        compute_viscosity_τII;
        relaxation = relaxation,
    )
    return nothing
end

@inline function update_viscosity!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F;
        relaxation = 1.0e0,
    ) where {F}

    fn = get_viscosity_fn(fn_viscosity)

    fn(
        stokes, phase_ratios, args, rheology, cutoff; relaxation = relaxation, air_phase = air_phase
    )
    return nothing
end

## 2D KERNELS

function compute_viscosity_τII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff, compute_viscosity_τII)
end

function compute_viscosity_εII!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    return compute_viscosity!(backend(stokes), stokes, relaxation, args, rheology, cutoff, compute_viscosity_εII)
end

# generic fallback
function compute_viscosity!(
        stokes::JustRelax.StokesArrays, args, rheology, cutoff; relaxation = 1.0e0
    )
    compute_viscosity_εII!(stokes, args, rheology, cutoff; relaxation = relaxation)
    return nothing
end

function compute_viscosity!(::CPUBackendTrait, stokes, ν, args, rheology, cutoff, fn_viscosity::F) where {F}
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff, fn_viscosity)
end

function _compute_viscosity!(stokes::JustRelax.StokesArrays, ν, args, rheology, cutoff, fn_viscosity::F) where {F}
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η, ν, @strain(stokes)..., args, rheology, cutoff, fn_viscosity
    )
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, Axx, Ayy, Axyv, args, rheology, cutoff, fn_viscosity::F
    ) where {F}
# Main.@infiltrate
    # convenience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)
    @inbounds begin
        # cache
        A = Axx[I...], Ayy[I...], Axyv[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(A...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        AII = second_invariant(AII_0 + A[1], -AII_0 + A[2], A[3])

        # compute and update stress viscosity
        ηi = fn_viscosity(rheology, AII, args_ij)
        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity_εII!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(η, ν, εII, args, rheology, cutoff, compute_viscosity_εII)
    return nothing
end

function compute_viscosity_τII!(η::AbstractArray, ν, εII::AbstractArray, args, rheology, cutoff)
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(η, ν, εII, args, rheology, cutoff, compute_viscosity_τII)
    return nothing
end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, AII, args, rheology, cutoff, fn_viscosity::F
    ) where {F}
    @inbounds begin
        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        AII_ij = AII[I...]

        # compute and update stress viscosity
        ηi = fn_viscosity(rheology, AII_ij, args_ij)

        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

function compute_viscosity_τII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff, compute_viscosity_τII
    )
    return nothing
end

function compute_viscosity_εII!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff, compute_viscosity_εII
    )
    return nothing
end

# fallback

function compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        phase_ratios,
        args,
        rheology,
        cutoff;
        air_phase::Integer = 0,
        relaxation = 1.0e0,
    )
    compute_viscosity!(
        backend(stokes), stokes, relaxation, phase_ratios, args, rheology, air_phase, cutoff, compute_viscosity_εII
    )
    return nothing
end


function compute_viscosity!(
        ::CPUBackendTrait,
        stokes::JustRelax.StokesArrays,
        ν,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    _compute_viscosity!(stokes, ν, args, rheology, air_phase, cutoff, fn_viscosity)

    return nothing
end

function compute_viscosity!(
        ::CPUBackendTrait,
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, air_phase, cutoff, fn_viscosity)

    return nothing
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        phase_ratios::JustPIC.PhaseRatios,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F
    ) where {F}
    ni = size(stokes.viscosity.η)
    # centered viscosity
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        phase_ratios.center,
        select_tensor_center(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args,
    )
    # vertex viscosity
    # skip for 3D for now, may change in the future
    if length(ni) == 2
        @parallel (@idx ni .+ 1) compute_viscosity_kernel!(
            stokes.viscosity.ηv,
            ν,
            phase_ratios.vertex,
            select_tensor_vertex(stokes, fn_viscosity)...,
            args,
            rheology,
            air_phase,
            cutoff,
            fn_viscosity,
            local_viscosity_args_vertex,
        )
    end
    return nothing
end

function _compute_viscosity!(
        stokes::JustRelax.StokesArrays,
        ν,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity::F,
        # do_vertices
    ) where {F}
    ni = size(stokes.viscosity.η)
    @parallel (@idx ni) compute_viscosity_kernel!(
        stokes.viscosity.η,
        ν,
        select_tensor_center(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        fn_viscosity,
        local_viscosity_args,
    )
    # skip for 3D for now, may change in the future
    length(size(phase_ratios.center)) == 3 && return

    @parallel (@idx ni .+ 1) compute_viscosity_kernel!(
        stokes.viscosity.ηv,
        ν,
        select_tensor_vertex(stokes, fn_viscosity)...,
        args,
        rheology,
        air_phase,
        cutoff,
        local_viscosity_args_vertex,
    )
    return nothing
end

for fn in (:select_tensor_center, :select_tensor_vertex)
    @eval @inline $fn(stokes, fn_viscosity) = $fn(stokes, fn_viscosity, JustRelax.static_dims(stokes))
end

# for 2D, we compute viscosity using the tensor defined at the cell centers or vertices, depending on the viscosity function
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_εII), ::Val{2}) = @strain_center(stokes)
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_τII), ::Val{2}) = @stress_center(stokes)
# in 3D we still do some interpolations
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_εII), ::Val{3}) = @strain(stokes)
@inline select_tensor_center(stokes, ::typeof(compute_viscosity_τII), ::Val{3}) = @stress(stokes)

# for 2D, we compute viscosity using the tensor defined at the cell centers or vertices, depending on the viscosity function
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_εII), ::Val{2}) = @tensor_vertex(stokes.ε)
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_τII), ::Val{2}) = @tensor_vertex(stokes.τ)
# in 3D we still do some interpolations
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_εII), ::Val{3}) = @strain(stokes.ε)
@inline select_tensor_vertex(stokes, ::typeof(compute_viscosity_τII), ::Val{3}) = @stress(stokes.τ)

@parallel_indices (I...) function compute_viscosity_kernel!(
        η, ν, ratios_center, Axx, Ayy, Axyv, args, rheology, air_phase::Integer, cutoff, fn_viscosity::F1, fn_args::F2
    ) where {F1, F2}

    # convenience closure
    Base.@propagate_inbounds @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # cache
        A = Axx[I...], Ayy[I...], Axyv[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(A...) * eps()

        # argument fields at local index
        args_ij = fn_args(args, I...)
        # args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        if air_phase > 0
            ratio_ij = correct_phase_ratio(air_phase, ratio_ij)
        end

        # compute second invariant of strain rate tensor
        Aij = AII_0 + A[1], -AII_0 + A[2], A[3]
        AII = second_invariant(Aij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)
        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## 3D KERNELS
# @parallel_indices (I...) function compute_viscosity_kernel!(
#         η, ν, Axx, Ayy, Azz, Ayzv, Axzv, Axyv, args, rheology, cutoff, fn_viscosity::F1, fn_args::F2
#     ) where {F1, F2}

#     # convenience closures
#     Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
#     Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
#     Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

#     @inbounds begin
#         Aij_normal = Axx[I...], Ayy[I...], Azz[I...]

#         # we need strain rate not to be zero, otherwise we get NaNs
#         AII_0 = allzero(Aij_normal...) * eps()

#         # # argument fields at local index
#         args_ijk = fn_args(args, I...)

#         # compute second invariant of strain rate tensor
#         Aij_normal = Aij_normal .+ (AII_0, -AII_0 * 0.5, -AII_0 * 0.5)
#         Aij_shear = gather_yz(Ayzv), gather_xz(Axzv), gather_xy(Axyv)
#         Aij = (Aij_normal..., Aij_shear...)
#         AII = second_invariant(Aij...)

#         # update stress and effective viscosity
#         ηi = fn_viscosity(rheology, AII, args_ijk)
#         ηi = continuation_linear(ηi, η[I...], ν)
#         η[I...] = clamp(ηi, cutoff...)
#     end

#     return nothing
# end

@parallel_indices (I...) function compute_viscosity_kernel!(
        η,
        ν,
        ratios_center,
        Axx,
        Ayy,
        Azz,
        Ayzv,
        Axzv,
        Axyv,
        args,
        rheology,
        air_phase::Integer,
        cutoff,
        fn_viscosity::F1,
        fn_args::F2
    ) where {F1, F2}

    # convenience closures
    Base.@propagate_inbounds @inline gather_yz(A) = _gather_yz(A, I...)
    Base.@propagate_inbounds @inline gather_xz(A) = _gather_xz(A, I...)
    Base.@propagate_inbounds @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        Aij_normal = Axx[I...], Ayy[I...], Azz[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        AII_0 = allzero(Aij_normal...) * eps()

        # # argument fields at local index
        args_ijk = fn_args(args, I...)

        # local phase ratio
        ratio_ijk = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ijk = correct_phase_ratio(air_phase, ratio_ijk)

        # compute second invariant of strain rate tensor
        Aij_normal = Aij_normal .+ (AII_0, -AII_0 * 0.5, -AII_0 * 0.5)
        Aij_shear = gather_yz(Ayzv), gather_xz(Axzv), gather_xy(Axyv)
        Aij = (Aij_normal..., Aij_shear...)
        AII = second_invariant(Aij...)

        # update stress and effective viscosity
        ηi = compute_phase_viscosity(rheology, ratio_ijk, AII, fn_viscosity, args_ijk)
        ηi = continuation_linear(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## HELPER FUNCTIONS

@inline function local_viscosity_args(args, I::Vararg{Integer, N}) where {N}
    v = map(values(args)) do a
        a isa AbstractArray ? getindex(a, I...) : a
    end
    local_args = (; zip(keys(args), v)..., dt = Inf, τII_old = 0.0)
    return local_args
end

@inline function local_viscosity_args_vertex(args, i, j)
    # clamp/wrap indices
    nx, ny = size(args[1])
    if PERIODIC_X_VISCOSITY_ARGS[]
        # periodic-x mapping from vertex index i -> neighboring center columns
        ir = mod1(i, nx)
        il = mod1(i - 1, nx)
    else
        il = max(i - 1, 1)  # left
        ir = min(i, nx)   # right
    end
    jb = max(j - 1, 1)  # bottom
    jt = min(j, ny)   # top
    # average values at cell centers surrounding vertex
    vals = values(args)
    v = map(vals) do a
        if a isa AbstractArray
            v11 = getindex(a, il, jb)
            v12 = getindex(a, ir, jb)
            v21 = getindex(a, il, jt)
            v22 = getindex(a, ir, jt)
            0.25 * (v11 + v12 + v21 + v22)
        else
            a
        end
    end
    # create local args
    local_args = (; zip(keys(args), v)..., dt = Inf, τII_old = 0.0)
    return local_args
end

@inline function local_viscosity_args_vertex(args, i, j, k)
    # clamp/wrap indices
    nx, ny, nz = size(args[1])
    if PERIODIC_X_VISCOSITY_ARGS[]
        ir = mod1(i, nx)
        il = mod1(i - 1, nx)
    else
        il = max(i - 1, 1)  # left
        ir = min(i, nx)   # right
    end
    jb = max(j - 1, 1)  # bottom
    jt = min(j, ny)   # top
    kf = max(k - 1, 1)  # front
    kb = min(k, nz)   # back
    # average values at cell centers surrounding vertex
    vals = values(args)
    v = map(vals) do a
        if a isa AbstractArray
            v111 = getindex(a, il, jb, kf)
            v121 = getindex(a, ir, jb, kf)
            v211 = getindex(a, il, jt, kf)
            v221 = getindex(a, ir, jt, kf)
            v112 = getindex(a, il, jb, kb)
            v122 = getindex(a, ir, jb, kb)
            v212 = getindex(a, il, jt, kb)
            v222 = getindex(a, ir, jt, kb)
            0.125 * (v111 + v121 + v211 + v221 + v112 + v122 + v212 + v222)
        else
            a
        end
    end
    # create local args
    local_args = (; zip(keys(args), v)..., dt = Inf, τII_old = 0.0)
    return local_args
end

@inline function local_args(args, I::Vararg{Integer, N}) where {N}
    v = map(values(args)) do a
        a isa AbstractArray ? getindex(a, I...) : a
    end
    local_args = (; zip(keys(args), v)..., dt = Inf, τII_old = 0.0)
    return local_args
end

@generated function compute_phase_viscosity(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, AII, fn_viscosity::F, args
    ) where {N, F}
    return quote
        @inline
        # Early exit: if single phase dominates (ratio ≈ 1), skip harmonic mean
        Base.@nexprs $N i -> begin
            if ratio[i] > 0.999  # faster than ≈ comparison
                return fn_viscosity(rheology[i].CompositeRheology[1], AII, args)
            end
        end

        η = 0.0
        Base.@nexprs $N i -> begin
            if !iszero(ratio[i])
                η += inv(fn_viscosity(rheology[i].CompositeRheology[1], AII, args)) * ratio[i]
            end
        end
        inv(η)
    end
end

# @generated function compute_phase_viscosity_εII(
#         rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, εII::T, args
#     ) where {N, T}
#     return quote
#         Base.@_inline_meta
#         η = zero(T)
#         Base.@nexprs $N i -> (
#             η += if iszero(ratio[i])
#                 zero(T)
#             else
#                 compute_viscosity_εII(rheology[i].CompositeRheology[1], εII, args) * ratio[i]
#             end
#         )
#         return η
#     end
# end

function correct_phase_ratio(air_phase, ratio::SVector{N, T}) where {N, T}
    if iszero(air_phase)
        return ratio
    elseif ratio[air_phase] ≈ 1
        return zeros(SVector{N, T})
    else
        mask = ntuple(i -> (i !== air_phase), Val(N))
        # set air phase ratio to zero
        corrected_ratio = ratio .* mask
        # normalize phase ratios without air
        return corrected_ratio ./ sum(corrected_ratio)
    end
end
