## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

"""Wrap active particle x-coordinates into `[xvi[1][1], xvi[1][end])`."""
function wrap_particles_x!(particles, xvi)
    xmin = xvi[1][1]
    xmax = xvi[1][end]
    lx = xmax - xmin
    ppx = particles.coords[1].data
    idx = particles.index.data

    @inbounds for k in eachindex(idx)
        if idx[k]
            x = ppx[k]
            if x < xmin
                ppx[k] = x + lx
            elseif x >= xmax
                ppx[k] = x - lx
            end
        end
    end
    return nothing
end

## END OF HELPER FUNCTION ------------------------------------------------------------
