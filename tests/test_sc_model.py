"""Sanity tests for sc_model.py. Run with: pytest tests/"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sc_model as m  # noqa: E402


def simulate_metrics(inputs, c, T, reps=200, seed=1, ss=0):
    """Pooled Bullwhip and NSAmp over many replications."""
    W = m.warmup_days(inputs, c)
    demand, lead = m.draw_randomness(inputs, W + inputs.horizon, reps, seed)
    out = m._simulate_core(inputs, c, np.full(reps, float(T)), ss, demand, lead)
    return m._block_metrics(out["order"][:, W:], out["net_stock"][:, W:], out["demand"][:, W:], c)


# 1. Deterministic steady state for every cadence -----------------------------

@pytest.mark.parametrize("c", [1, 7, 30])
@pytest.mark.parametrize("mu_L", [7, 10])
@pytest.mark.parametrize("T", [1.0, 2.5])
def test_deterministic_steady_state(c, mu_L, T):
    inputs = m.SCInputs(mu_d=50, sigma_d=0, mu_L=mu_L, sigma_L=0)
    ss = 20
    df = m.simulate_run(inputs, c, ss=ss, k=50, T=T)

    review = df[df["day"] % c == 0]
    assert (review["order"] == c * inputs.mu_d).all()
    assert (df.loc[df["day"] % c != 0, "order"] == 0).all()

    arrival_days = df.index[df["arrivals"] > 0]
    arrival_days = arrival_days[arrival_days > 0]
    assert len(arrival_days) > 0
    assert np.allclose(df.loc[arrival_days - 1, "net_stock"], ss)
    assert (df["backlog"] == 0).all()
    assert df["fill_rate_cumulative"].iloc[-1] == 1.0


# 2. Bullwhip = 1/(2T - 1) ------------------------------------------------------

@pytest.mark.parametrize("sigma_L", [0.0, 2.0])
@pytest.mark.parametrize("T", [1.0, 2.0, 5.0])
def test_bullwhip_matches_theory(T, sigma_L):
    inputs = m.SCInputs(mu_d=100, sigma_d=10, mu_L=8, sigma_L=sigma_L)
    bullwhip, _ = simulate_metrics(inputs, 1, T)
    assert bullwhip == pytest.approx(1 / (2 * T - 1), rel=0.05)


# 3. NSAmp = 1 + mu_L at T = 1 ---------------------------------------------------
# Convention: an order placed at the end of day t is received at the start of
# day t+L+1, and net stock is measured at the end of the day. Net stock then
# carries L+1 days of demand uncertainty, so NSAmp = 1 + L exactly (not L).

@pytest.mark.parametrize("mu_L", [5, 10])
def test_nsamp_order_up_to(mu_L):
    inputs = m.SCInputs(mu_d=100, sigma_d=10, mu_L=mu_L, sigma_L=0)
    _, nsamp = simulate_metrics(inputs, 1, 1.0)
    assert nsamp == pytest.approx(1 + mu_L, rel=0.05)


# 4. Empirical k* matches theory --------------------------------------------------

@pytest.mark.parametrize("c", [1, 7])
@pytest.mark.parametrize("T", [1.0, 3.0])
def test_capacity_matches_theory(T, c):
    inputs = m.SCInputs()
    W = m.warmup_days(inputs, c)
    demand, lead = m.draw_randomness(inputs, W + inputs.horizon, 100, 7)
    row = m._evaluate_rows(inputs, c, [T], demand, lead, W)[0]
    th = m.theory(inputs, c, T)
    assert row["k_star_T"] == pytest.approx(th["k_theory"], abs=0.1 * th["sigma_P"] + 0.1)


@pytest.mark.parametrize("u, w", [(2.0, 3.0), (2.0, 4.0), (2.0, 7.0)])
def test_capacity_side_of_median_and_brute_force(u, w):
    # Capacity cost u*k + w*max(0, P - k): k* is the (1 - u/w) quantile, so it
    # sits below median production when w < 2u and above it when w > 2u.
    inputs, c, T = m.SCInputs(u=u, w=w), 7, 2.0
    W = m.warmup_days(inputs, c)
    demand, lead = m.draw_randomness(inputs, W + inputs.horizon, m.N_REPS, 3)
    k_star = m._evaluate_rows(inputs, c, [T], demand, lead, W)[0]["k_star_T"]
    prod = m._simulate_core(inputs, c, np.full(m.N_REPS, T), 0, demand, lead)["production"][:, W:]

    median = np.median(prod)
    if w < 2 * u:
        assert k_star < median
    elif w > 2 * u:
        assert k_star > median
    else:
        assert k_star == pytest.approx(median, abs=0.05)

    k_grid = np.arange(40, 60, 0.01)
    costs = [m._capacity_cost(prod, k, u, w).sum(axis=1).mean() for k in k_grid]
    best = m._capacity_cost(prod, k_star, u, w).sum(axis=1).mean()
    assert best <= min(costs) + 1e-6


def test_theory_unstable_returns_none():
    th = m.theory(m.SCInputs(), 1, 0.5)
    assert th["k_theory"] is None and th["bullwhip_theory"] is None


# 5. Fill rate is non-decreasing in ss ---------------------------------------------

@pytest.mark.parametrize("c", [1, 7, 30])
def test_fill_rate_monotone_in_ss(c):
    inputs = m.SCInputs()
    fills = [m.evaluate_params(inputs, c, ss, 50, 2.0, reps=10)["fill_rate"]
             for ss in range(-150, 301, 25)]
    assert all(np.diff(fills) >= 0)
    assert fills[-1] > fills[0]


# 6. Optimiser result is feasible and no feasible grid T beats it --------------------

@pytest.fixture(scope="module", params=[1, 7, 30])
def opt(request):
    inputs = m.SCInputs()
    return inputs, request.param, m.optimise(inputs, request.param)


def test_optimum_feasible_and_best_on_grid(opt):
    inputs, c, r = opt
    assert r.fill_rate >= inputs.fill_target
    assert m.T_MIN_OPT <= r.T_star <= m.T_MAX_OPT
    grid = r.sweep[(r.sweep["T"] >= m.T_MIN_OPT) & (r.sweep["fill_rate"] >= inputs.fill_target)]
    assert (grid["total_cost"] >= r.cost - 1e-6).all()


def test_optimum_reproduced_by_resimulation(opt):
    inputs, c, r = opt
    ev = m.evaluate_params(inputs, c, r.ss_star, r.k_star, r.T_star, seed=r.seed)
    assert ev["total_cost"] == pytest.approx(r.cost, rel=1e-9)
    assert ev["fill_rate"] == pytest.approx(r.fill_rate, rel=1e-9)


def test_order_sequence_independent_of_ss():
    inputs = m.SCInputs()
    a = m.simulate_run(inputs, 7, ss=0, k=50, T=2.0)
    b = m.simulate_run(inputs, 7, ss=150, k=50, T=2.0)
    assert (a["order"] == b["order"]).all()
    assert np.allclose(b["net_stock"] - a["net_stock"], 150)


# 7. Brute-force cross-check with differential evolution ------------------------------

def test_matches_differential_evolution():
    inputs, c, seed = m.SCInputs(), 7, 42
    r = m.optimise(inputs, c, seed=seed)

    W = m.warmup_days(inputs, c)
    demand, lead = m.draw_randomness(inputs, W + inputs.horizon, m.N_REPS, seed)

    @lru_cache(maxsize=None)
    def sim(T):
        out = m._simulate_core(inputs, c, np.full(m.N_REPS, T), 0, demand, lead)
        return (out["net_stock"][:, W:], out["net_stock_prev"][:, W:],
                out["arrivals"][:, W:], out["demand"][:, W:], out["production"][:, W:])

    def objective(x):
        ss, k, T = int(round(x[0])), x[1], round(x[2], 3)
        ns0, ns_prev0, arr, dem, prod = sim(T)
        cost = m._mean_total_cost(ns0, prod, ss, k, inputs)
        shortfall = max(0.0, inputs.fill_target - m._fill_rate(ns_prev0, arr, dem, ss))
        return cost * (1 + 100 * shortfall)

    de = differential_evolution(objective, bounds=[(-200, 400), (30, 90), (1, 10)],
                                seed=0, popsize=15, maxiter=60, tol=1e-8, polish=False)
    assert de.fun == pytest.approx(r.cost, rel=0.02)
