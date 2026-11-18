# Scalar stick-slip cycle for the SEAS BP1-FD velocity-weakening numbers.
# The DYREL strip that uses the same parameters is
# miniapps/QuakeSystem/SEAS_stickslip_eigenstrain.jl.

using Test
using JustRelax

include(joinpath(@__DIR__, "..", "miniapps", "QuakeSystem", "seas_stickslip_slider.jl"))

const _YEAR = 365.25 * 86400.0

@testset "SEAS velocity-weakening slider cycles" begin
    p = seas_bp1_vw_params()
    # A single fault cell is stiffer than critical, so it only creeps.
    # The 4 km spring is not.
    @test p.k_fault > p.k_crit
    @test p.k < p.k_crit
    @test p.V > 0
    @test p.V < p.Vp

    hist = integrate_slider(p; nsteps = 30_000, max_time = 50 * _YEAR, dt_floor = 1.0, max_events = 1)
    summary = summarize_stickslip(hist)
    @test summary.n >= 1
    @test summary.drops[1] > 1.0e6
    @test 1.0e-2 < summary.vmax[1] < 10.0
    @test all(s -> isfinite(s.τ) && isfinite(s.ψ) && isfinite(s.V) && s.V >= 0, hist)
    # The cycle is closed: the run stops on the step that locks again.
    @test hist[end].V <= 1.0e-9
    @test any(s -> s.V > 1.0e-6, hist)
end
