abstract type AbstractBoundaryConditions end
abstract type AbstractFlowBoundaryConditions <: AbstractBoundaryConditions end

@inline _bc_value(bc, key::Symbol) = hasproperty(bc, key) ? getproperty(bc, key) : false
@inline function _thermal_bc_tuple(bc, ::Val{2})
    return (
        left = _bc_value(bc, :left),
        right = _bc_value(bc, :right),
        top = _bc_value(bc, :top),
        bot = _bc_value(bc, :bot),
    )
end
@inline function _thermal_bc_tuple(bc, ::Val{3})
    return (
        left = _bc_value(bc, :left),
        right = _bc_value(bc, :right),
        front = _bc_value(bc, :front),
        back = _bc_value(bc, :back),
        top = _bc_value(bc, :top),
        bot = _bc_value(bc, :bot),
    )
end

"""
    TemperatureBoundaryConditions(; no_flux, constant_flux, constant_value, periodic, dirichlet)

Create thermal boundary conditions for 2D or 3D temperature fields.

Boundary tuples use `left`, `right`, `top`, and `bot` in 2D. In 3D they also use
`front` and `back`. Omitted faces are filled with `false`, and the dimensionality is
inferred from the longest boundary tuple that is passed.

The face values have the following meaning:

- `no_flux`: `true` copies the adjacent interior temperature into the ghost layer.
- `constant_value`: numeric values prescribe the boundary temperature through the
  ghost value `Tghost = 2 * value - Tinterior`.
- `constant_flux`: numeric values prescribe heat fluxes in the pseudo-transient
  diffusion flux kernels.
- `periodic`: `true` copies the opposite interior temperature into the ghost layer.
- `false`: leaves that boundary inactive for the corresponding condition.

`dirichlet` accepts the mask-based Dirichlet forms supported by `Dirichlet`, for
example `(; constant = value, mask = mask)`.

# Examples

```julia
TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, top = false, bot = false),
    constant_value = (top = 273.0, bot = 1573.0),
)

TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, front = true, back = true, top = false, bot = false),
    constant_flux = (top = 0.0, bot = 0.03),
    periodic = (left = false, right = false, front = false, back = false, top = false, bot = false),
)
```
"""
struct TemperatureBoundaryConditions{T1, T2, T3, T4, D, nD} <: AbstractBoundaryConditions
    no_flux::T1
    constant_flux::T2
    constant_value::T3
    periodic::T4
    dirichlet::D
    function TemperatureBoundaryConditions(;
            no_flux::T1 = (left = true, right = false, top = false, bot = false),
            constant_flux::T2 = (left = false, right = false, top = false, bot = false),
            constant_value::T3 = (left = false, right = false, top = false, bot = false),
            periodic::T4 = (left = false, right = false, top = false, bot = false),
            dirichlet = (; constant = nothing, mask = nothing),
        ) where {T1, T2, T3, T4}

        @inline get_dimension(::NTuple{4, Bool}) = 2
        @inline get_dimension(::NTuple{6, Bool}) = 3

        D = Dirichlet(dirichlet)
        nD = get_dimension(values(no_flux))

        # expand to 3D
        dummy = (; front = false, back = false)

        no_flux_exp = merge(dummy, no_flux)
        constant_flux_exp = merge(dummy, constant_flux)
        constant_value_exp = merge(dummy, constant_value)
        periodic_exp = merge(dummy, periodic)

        return new{typeof(no_flux_exp), typeof(constant_flux_exp), typeof(constant_value_exp), typeof(periodic_exp), typeof(D), nD}(
            no_flux_exp, constant_flux_exp, constant_value_exp, periodic_exp, D
        )
    end
end

struct DisplacementBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool

    function DisplacementBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            free_surface::Bool = false,
        ) where {T}
        @assert length(no_slip) === length(free_slip)
        check_flow_bcs(no_slip, free_slip)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T, nD}(no_slip, free_slip, free_surface)
    end
end
@inline _component_dirichlet(nt, key::Symbol) =
    Dirichlet(hasproperty(nt, key) ? getproperty(nt, key) : NamedTuple())

@inline function _velocity_dirichlet(nt, ::Val{2})
    return (Vx = _component_dirichlet(nt, :Vx), Vy = _component_dirichlet(nt, :Vy))
end
@inline function _velocity_dirichlet(nt, ::Val{3})
    return (
        Vx = _component_dirichlet(nt, :Vx),
        Vy = _component_dirichlet(nt, :Vy),
        Vz = _component_dirichlet(nt, :Vz),
    )
end

"""
    VelocityBoundaryConditions(; no_slip, free_slip, free_surface=false, periodic, dirichlet=NamedTuple(), prescribed=NamedTuple())

Define 2D or 3D boundary conditions for the velocity field. Face names are
`left`, `right`, `top`, and `bot` in 2D, with `front` and `back` added in 3D.

`no_slip` and `free_slip` must differ on every face that is not periodic.
Periodic faces must set both `no_slip` and `free_slip` to `false`. Left/right
periodicity must be enabled together (same for top/bot).

`periodic` wraps staggered velocity ghosts across opposite faces (same pattern
as temperature `periodic`, with staggered `Vx`/`Vy` indexing).

`prescribed` optionally sets wall values after face BCs. Supported keys:
- `top_Vx::Real`: Dirichlet top-wall tangential velocity via ghost fill
  `Vx[:,end] = 2*top_Vx - Vx[:,end-1]` and `Vy[:,end] = 0` (simple-shear drive).

`dirichlet` prescribes an interior, mask-selected Dirichlet region for the
velocity field (for example an internal "velocity box"), independent of the
four/six edge faces above. Pass a per-component named tuple, e.g.
`dirichlet = (; Vx = (; constant = v, mask = mask_x))`; a component that is
omitted, or the keyword itself, leaves that component unconstrained. Each
component's `mask` must be sized like that component's velocity array (`Vx`,
`Vy`, or `Vz`), not the interior-only residual array.

For a spatially-varying prescribed value (e.g. several boxes with different
velocities), build a `DirichletBoundaryCondition(value_array, Mask(mask_array))`
directly and pass it as the component, e.g. `dirichlet = (; Vx = my_bc)`: the
`(; constant, mask)` shorthand's array form infers its mask from the value
array's non-zero entries, which cannot represent a prescribed value of exactly
zero. See [`Dirichlet`](@ref).
"""
struct VelocityBoundaryConditions{T, P, D, Pr, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool
    periodic::P
    dirichlet::D
    prescribed::Pr

    function VelocityBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            free_surface::Bool = false,
            periodic::P = (left = false, right = false, top = false, bot = false),
            dirichlet::NamedTuple = NamedTuple(),
            prescribed::Pr = NamedTuple(),
        ) where {T, P, Pr}
        @assert length(no_slip) === length(free_slip)
        @assert length(no_slip) === length(periodic)
        check_periodic_pairs(periodic)
        check_flow_bcs(no_slip, free_slip, periodic)

        nD = length(no_slip) == 4 ? 2 : 3
        # expand to 3D face names when needed (matches TemperatureBoundaryConditions)
        dummy = (; front = false, back = false)
        periodic_exp = merge(dummy, periodic)
        D_nt = _velocity_dirichlet(dirichlet, Val(nD))
        return new{T, typeof(periodic_exp), typeof(D_nt), Pr, nD}(
            no_slip, free_slip, free_surface, periodic_exp, D_nt, prescribed
        )
    end
end

function check_periodic_pairs(periodic)
    if getproperty(periodic, :left) != getproperty(periodic, :right)
        error("x-periodicity requires both `left` and `right` periodic=true")
    end
    if getproperty(periodic, :top) != getproperty(periodic, :bot)
        error("y-periodicity requires both `top` and `bot` periodic=true")
    end
    if hasproperty(periodic, :front) && hasproperty(periodic, :back)
        if getproperty(periodic, :front) != getproperty(periodic, :back)
            error("z-periodicity requires both `front` and `back` periodic=true")
        end
    end
    return nothing
end

function check_flow_bcs(no_slip::T, free_slip::T, periodic = nothing) where {T}
    v1 = values(no_slip)
    v2 = values(free_slip)
    k = keys(no_slip)
    for (v1, v2, k) in zip(v1, v2, k)
        is_periodic = !isnothing(periodic) && getproperty(periodic, k)
        if is_periodic
            (v1 || v2) && error(
                "Incompatible BCs: periodic `$k` cannot be combined with no_slip/free_slip on that face",
            )
            continue
        end
        if v1 == v2
            error(
                "Incompatible boundary conditions. The $k boundary condition can't be the same for no_slip and free_slip",
            )
        end
    end
    return
end
