# Integration test: same coupled 0D path as `RheologyCalculator.jl/examples/RateState_compare_JustRelax.jl`,
# shorter time horizon (~1000/3.5 steps) to keep CI light while spanning elastic loading and a stress drop.
# Asserts JR ≈ RC at five time indices (loose tolerances: different constitutive paths, same driver).
# Writes a quick diagnostic PNG under `tempdir()` (visual agreement is the primary sanity check).

using Test
using JustRelax
using RheologyCalculator

const RC_ROOT = joinpath(@__DIR__, "..", "RheologyCalculator.jl")
include(joinpath(RC_ROOT, "rheologies", "RheologyDefinitions.jl"))
include(joinpath(RC_ROOT, "rheologies", "RateState_HypoPlastic.jl"))
include(joinpath(@__DIR__, "ratestate_0d_coupled.jl"))

"""Five indices spread over [2, ntime] (skip i=1 where Ω/V are not advanced in the coupled loop)."""
function snapshot_indices(ntime::Int)
    @assert ntime >= 8
    return Int[
        2,
        max(3, round(Int, 0.2 * ntime)),
        max(4, round(Int, 0.45 * ntime)),
        max(5, round(Int, 0.7 * ntime)),
        ntime,
    ]
end

@testset "RSF 0D: RC vs JR snapshots (short coupled march)" begin
    hyp, elastic, c, x, xnorm, vars, others = default_ratestate_series_setup()
    ntime = max(8, round(Int, 1000 / 3.5))
    dt0 = 1e6

    t_v, τ_rc, Ω_rc, V_rc, τy_rc, τ_jr, Ω_jr, V_jr, τy_jr =
        stress_time_coupled(c, vars, x, xnorm, others, hyp, elastic; ntime = ntime, dt0 = dt0)

    idx = snapshot_indices(ntime)
    @test issorted(idx)
    @test idx[1] >= 2
    @test idx[end] == ntime

    # JR uses the same dt, Ω_old, and τ memory as RC, but RC still solves the full series
    # balance each step while JR is a local Maxwell+RSF update — O(0.1–1)% (sometimes a few
    # % on Vp) drift after O(10²) steps is normal. The plot is the right check for “aligned”;
    # these bounds reject gross regression without demanding bitwise parity with `solve`.
    rtol_τ = 2.5e-2
    atol_τ = 2.0e5
    rtol_Ω = 2.5e-2
    atol_Ω = 6.0e-2
    rtol_V = 1.0e-1
    atol_V = 1.0e-15
    rtol_y = 2.5e-2
    atol_y = 2.0e5

    for i in idx
        @test isapprox(τ_jr[i], τ_rc[i]; rtol = rtol_τ, atol = atol_τ)
        @test isapprox(Ω_jr[i], Ω_rc[i]; rtol = rtol_Ω, atol = atol_Ω)
        @test isapprox(V_jr[i], V_rc[i]; rtol = rtol_V, atol = atol_V)
        @test isapprox(τy_jr[i], τy_rc[i]; rtol = rtol_y, atol = atol_y)
    end

    relτ = maximum(abs.(τ_jr[idx] .- τ_rc[idx]) ./ max.(abs.(τ_rc[idx]), 1.0))
    @test relτ < 0.03

    # Diagnostic figure (overwrites each run): `tempdir()/justrelax_rsf_0d_snapshot_test.png`
    plot_path = joinpath(tempdir(), "justrelax_rsf_0d_snapshot_test.png")
    try
        using Plots
        SecYear = 3600 * 24 * 365.25
        tx = t_v ./ SecYear
        tx2 = t_v[2:end] ./ SecYear
        pτ = plot(
            tx,
            τy_rc ./ 1e6;
            lab = "tau_interface (RC)",
            lc = :red,
            lw = 2,
            title = "RC vs JR (short 0D test)",
            xlabel = "t [yr]",
            ylabel = "tauII [MPa]",
        )
        plot!(pτ, tx, τ_rc ./ 1e6; lab = "tauII (RC)", lc = :blue, lw = 2)
        plot!(pτ, tx, τy_jr ./ 1e6; lab = "tau_interface (JR)", lc = :magenta, ls = :dash, lw = 2)
        plot!(pτ, tx, τ_jr ./ 1e6; lab = "tauII (JR)", lc = :cyan, ls = :dash, lw = 2)
        scatter!(
            pτ,
            t_v[idx] ./ SecYear,
            τ_rc[idx] ./ 1e6;
            lab = "snapshots",
            ms = 4,
            mc = :black,
        )

        pΩ = plot(tx2, Ω_rc[2:end]; lab = "Omega (RC)", lc = :black, lw = 2, xlabel = "t [yr]", ylabel = "Omega")
        plot!(pΩ, tx2, Ω_jr[2:end]; lab = "Omega (JR)", lc = :gray, ls = :dash, lw = 2)
        pΩr = twinx(pΩ)
        plot!(pΩr, tx2, V_rc[2:end]; yscale = :log10, lab = "Vp (RC)", lc = :green, lw = 2)
        plot!(pΩr, tx2, V_jr[2:end]; yscale = :log10, lab = "Vp (JR)", lc = :olive, ls = :dash, lw = 2)
        plot!(pΩr, ylabel = "Vp [m/s]")

        p = plot(pτ, pΩ; layout = (1, 2), size = (1100, 420))
        savefig(p, plot_path)
        @info "RSF 0D snapshot diagnostic plot saved" path = plot_path snapshot_indices = idx ntime = ntime
    catch e
        @warn "Could not write diagnostic plot (Plots unavailable or error)." exception = (e,) path = plot_path
    end
end
