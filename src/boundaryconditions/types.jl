abstract type AbstractBoundaryConditions end
abstract type AbstractFlowBoundaryConditions <: AbstractBoundaryConditions end
struct TemperatureBoundaryConditions{T, D, nD} <: AbstractBoundaryConditions
    no_flux::T
    dirichlet::D
    periodic_x::Bool
    function TemperatureBoundaryConditions(;
            no_flux::T = (left = true, right = false, top = false, bot = false),
            dirichlet = (; constant = nothing, mask = nothing),
            periodic_x::Bool = false,
        ) where {T}
        D = Dirichlet(dirichlet)
        nD = length(no_flux) == 4 ? 2 : 3
        check_thermal_bcs(no_flux, periodic_x)
        return new{T, typeof(D), nD}(no_flux, D, periodic_x)
    end
end

struct DisplacementBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool
    periodic_x::Bool

    function DisplacementBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            free_surface::Bool = false,
            periodic_x::Bool = false,
        ) where {T}
        @assert length(no_slip) === length(free_slip)
        check_flow_bcs(no_slip, free_slip, periodic_x)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T, nD}(no_slip, free_slip, free_surface, periodic_x)
    end
end
struct VelocityBoundaryConditions{T, nD} <: AbstractFlowBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool
    periodic_x::Bool

    function VelocityBoundaryConditions(;
            no_slip::T = (left = false, right = false, top = false, bot = false),
            free_slip::T = (left = true, right = true, top = true, bot = true),
            free_surface::Bool = false,
            periodic_x::Bool = false,
        ) where {T}
        @assert length(no_slip) === length(free_slip)
        check_flow_bcs(no_slip, free_slip, periodic_x)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T, nD}(no_slip, free_slip, free_surface, periodic_x)
    end
end

function check_flow_bcs(no_slip::T, free_slip::T, periodic_x::Bool = false) where {T}
    v1 = values(no_slip)
    v2 = values(free_slip)
    k = keys(no_slip)
    for (v1, v2, k) in zip(v1, v2, k)
        if periodic_x && (k == :left || k == :right)
            continue
        end
        if v1 == v2
            error(
                "Incompatible boundary conditions. The $k boundary condition can't be the same for no_slip and free_slip",
            )
        end
    end
    if periodic_x
        (getproperty(no_slip, :left) || getproperty(no_slip, :right)) &&
            error("Incompatible BCs: `periodic_x=true` cannot be combined with left/right no-slip.")
        (getproperty(free_slip, :left) || getproperty(free_slip, :right)) &&
            error("Incompatible BCs: `periodic_x=true` cannot be combined with left/right free-slip.")
    end
    return
end

function check_thermal_bcs(no_flux::T, periodic_x::Bool) where {T}
    if periodic_x && (getproperty(no_flux, :left) || getproperty(no_flux, :right))
        error("Incompatible thermal BCs: `periodic_x=true` cannot be combined with left/right no-flux.")
    end
    return nothing
end
