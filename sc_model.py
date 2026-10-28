"""
Supply chain simulation, optimisation and theory for the Streamlit app.

A single manufacturer under normal conditions (no disruptions) ordering with a
Proportional Order Up-To (POUT) policy on a review cadence c (1, 7 or 30 days).
Pure numpy / scipy / pandas: no Streamlit imports.

=============================================================================
DIFFERENCES FROM supply_chain_sim.py (the earlier DQN project)
=============================================================================
 1. Order equation input: compute_order() is passed NET STOCK (inventory -
    backlog) instead of on-hand inventory, so backlogs drive orders.
 2. Order timing: the order is placed at the END of the day, after arrivals,
    demand and fulfilment. supply_chain_sim.py computes it mid-step using the
    previous day's inventory.
 3. Lead-time convention: an order placed at the end of day t with lead time L
    spends L full days in transit (days t+1 .. t+L) and is received at the start
    of day t+L+1. supply_chain_sim.py receives it at the start of day t+L.
    With this convention the deterministic steady state has actual WIP equal to
    desired WIP (mu_L * mu_d) for daily review, net stock just before every
    arrival equal to ss for every cadence, and NSAmp = 1 + L at T = 1
    (the textbook "1 + Tp" result).
 4. actual_wip is the exact sum of orders placed but not yet received (tracked
    as a running total). supply_chain_sim.py uses a deque of the last
    LEAD_TIME orders, which includes the order arriving that day and ignores
    random lead times.
 5. Generalised for cadence: demand_forecast = c * mu_d, desired_wip =
    mu_L * mu_d (mean lead time input, not the LEAD_TIME constant), and orders
    are placed only on review days (t % c == 0). supply_chain_sim.py orders
    every day.
 6. One supplier, one route: no alpha split, no capacity factors C^A / C^B,
    no traffic, no disruption scenarios. Lead time is
    max(1, round(Normal(mu_L, sigma_L))) instead of 5 + N(0, 1 or 2).
 7. Fulfilment: from on-hand stock (yesterday's inventory + today's
    arrivals), first today's demand, then outstanding backlog.
    supply_chain_sim.py uses Available = I + Q - B and F = min(Available, D),
    which serves backlog first. Our "available" column is on-hand stock
    (I_{t-1} + Q_t), not I + Q - B.
 8. Fill rate is sum(fulfilled_immediate) / sum(demand) after warm-up, not
    the per-step F_t / D_t.
 9. NSAmp = Var(net stock at review points) / Var(demand aggregated over c
    days) and Bullwhip = Var(orders per review) / Var(aggregated demand).
    supply_chain_sim.py uses Var(on-hand inventory) / Var(daily demand),
    running from step 1.
10. Initial conditions are the exact deterministic steady state rather than
    inventory = 29 with a pipeline of mu per day. Net stock at the end of day 0
    is ss + mu_d * (round(mu_L) mod c): exactly ss whenever c divides
    round(mu_L) (always for daily review). The extra term is the steady-state
    offset that appears when the lead time is not a multiple of the cadence;
    it does not depend on ss, so the order sequence is independent of ss.
11. Randomness uses numpy default_rng with separate child streams for demand
    and lead times (common random numbers across every T, ss and k evaluated
    in one optimisation). supply_chain_sim.py seeds the global np.random.
12. Orders are capped at ORDER_CAP_FACTOR * c * mu_d. This never binds for
    stable T (> 0.5); it only stops unstable runs (T <= 0.5) overflowing.
13. New: production smoothing, capacity, the cost model, warm-up, theory
    and optimisation. Capacity cost per day is u*k + w*max(0, P - k): every
    unit of capacity costs u whether used or not, and every unit produced
    above capacity costs w (overtime).
=============================================================================
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


CADENCES = {"Daily": 1, "Weekly": 7, "Monthly": 30}
N_REPS = 30
T_GRID = np.round(np.arange(1, 101) * 0.1, 1)     # 0.1, 0.2, ..., 10.0
T_MIN_OPT, T_MAX_OPT = 1.0, 10.0
REFINE_HALF_WIDTH = 0.2
ORDER_CAP_FACTOR = 1000
T_CHUNK = 25                                      # T values simulated per batch


@dataclass(frozen=True)
class SCInputs:
    mu_d: float = 50.0          # mean daily demand
    sigma_d: float = 10.0       # demand standard deviation
    mu_L: float = 7.0           # mean lead time (days)
    sigma_L: float = 2.0        # lead time standard deviation
    fill_target: float = 0.95   # target fill rate
    h: float = 1.0              # holding cost per unit per day
    b: float = 9.0              # backlog cost per unit per day
    u: float = 2.0              # cost per unit of capacity per day
    w: float = 4.0              # cost per unit of overtime production (w > u)
    horizon: int = 365          # reported days after warm-up


# -----------------------------------------------------------------------------
# Order equation (POUT), generalised for cadence and random lead times
# -----------------------------------------------------------------------------
# O_t = max(0, round(c*mu_d + (1/T)(ss - NS_t) + (1/T)(mu_L*mu_d - WIP_t)))
# With T = 1 this is order-up-to with S = mu_d*(mu_L + c) + ss.
# Works element-wise on numpy arrays as well as scalars.
# -----------------------------------------------------------------------------

def compute_order(net_stock, safety_stock, outstanding_orders, c, mu_d, mu_L, T):
    demand_forecast      = c * mu_d                     # mean demand over one review period
    inventory_correction = (1 / T) * (safety_stock - net_stock)
    desired_wip          = mu_L * mu_d                  # mean lead time (days) x daily mean
    actual_wip           = outstanding_orders           # running sum: placed but NOT yet received
    pipeline_correction  = (1 / T) * (desired_wip - actual_wip)
    return np.maximum(0, np.round(demand_forecast + inventory_correction + pipeline_correction))


# -----------------------------------------------------------------------------
# Horizon helpers and random draws
# -----------------------------------------------------------------------------

def warmup_days(inputs, c):
    """max(60, 3*mu_L) rounded up to a whole number of review periods, so the
    reporting window starts on a review day."""
    w = max(60, int(np.ceil(3 * inputs.mu_L)))
    return int(np.ceil(w / c) * c)


def draw_randomness(inputs, n_days, reps, seed):
    """Demand and lead-time draws, shape (reps, n_days). Separate child streams
    keep replication 0 identical whatever the number of replications."""
    demand_rng, lead_rng = (np.random.default_rng(s)
                            for s in np.random.SeedSequence(seed).spawn(2))
    demand = np.maximum(0, np.round(demand_rng.normal(inputs.mu_d, inputs.sigma_d, (reps, n_days))))
    lead = np.maximum(1, np.round(lead_rng.normal(inputs.mu_L, inputs.sigma_L, (reps, n_days)))).astype(np.int64)
    return demand, lead


# -----------------------------------------------------------------------------
# Vectorised simulation core
# -----------------------------------------------------------------------------
# Rows are independent runs (replication x T value). The only Python loop is
# over days. Daily order of events:
#   1. receive arrivals   2. demand   3. fulfil today's demand, then backlog
#   4. update net stock   5. on review days place an order
# -----------------------------------------------------------------------------

def _simulate_core(inputs, c, T_rows, ss, demand, lead):
    M, N = demand.shape
    mu_d, mu_L = inputs.mu_d, inputs.mu_L
    L0 = int(round(mu_L))
    cap = ORDER_CAP_FACTOR * c * mu_d + 1

    arrivals = np.zeros((M, N + int(lead.max()) + 2))
    production = np.zeros((M, N + c + 1))

    # Deterministic steady state: orders of c*mu_d placed on days -c, -2c, ...
    # still in transit at the start of day 0 (arrival day = placed + L0 + 1).
    j = 1
    while -j * c + L0 + 1 >= 0:
        arrivals[:, -j * c + L0 + 1] += c * mu_d
        j += 1
    wip = arrivals[:, :L0 + 2].sum(axis=1)
    ns_prev = np.full(M, ss + mu_d * ((L0 + 1) % c), dtype=float)
    production[:, 0] = mu_d

    out = {k: np.zeros((M, N)) for k in
           ("arrivals", "available", "fulfilled_immediate", "fulfilled_backlog",
            "net_stock", "order", "wip")}
    out["lead_time_of_order"] = np.full((M, N), np.nan)
    rows = np.arange(M)

    for t in range(N):
        arr = arrivals[:, t]
        wip -= arr
        inv_prev = np.maximum(ns_prev, 0)
        backlog_prev = np.maximum(-ns_prev, 0)
        available = inv_prev + arr
        d = demand[:, t]
        fi = np.minimum(d, available)
        fb = np.minimum(backlog_prev, available - fi)
        ns = ns_prev + arr - d

        if t % c == 0:
            o = np.minimum(compute_order(ns, ss, wip, c, mu_d, mu_L, T_rows), cap)
            lt = lead[:, t]
            arrivals[rows, t + lt + 1] += o
            wip += o
            production[:, t + 1:t + 1 + c] += (o / c)[:, None]
            out["order"][:, t] = o
            out["lead_time_of_order"][:, t] = lt

        out["arrivals"][:, t] = arr
        out["available"][:, t] = available
        out["fulfilled_immediate"][:, t] = fi
        out["fulfilled_backlog"][:, t] = fb
        out["net_stock"][:, t] = ns
        out["wip"][:, t] = wip
        ns_prev = ns

    out["demand"] = demand
    out["production"] = production[:, :N]
    # net stock at the end of the previous day (needed to re-price ss shifts)
    out["net_stock_prev"] = np.concatenate(
        [np.full((M, 1), ss + mu_d * ((L0 + 1) % c)), out["net_stock"][:, :-1]], axis=1)
    return out


# -----------------------------------------------------------------------------
# Metrics and costs
# -----------------------------------------------------------------------------

def _block_metrics(orders, net_stock, demand, c):
    """Bullwhip and NSAmp at review-period level. Arrays are post-warm-up,
    shape (reps, days), starting on a review day. Variances are pooled."""
    n_blocks = demand.shape[1] // c
    agg = demand[:, :n_blocks * c].reshape(demand.shape[0], n_blocks, c).sum(axis=2)
    var_d = np.var(agg)
    if var_d == 0:
        return np.nan, np.nan
    review = np.arange(n_blocks) * c
    return np.var(orders[:, review]) / var_d, np.var(net_stock[:, review]) / var_d


def _fill_rate(ns_prev0, arrivals, demand, ss):
    """Mean over replications of sum(fulfilled_immediate) / sum(demand), for
    net stock NS(ss) = NS(0) + ss."""
    fi = np.minimum(demand, np.maximum(ns_prev0 + ss, 0) + arrivals)
    return float(np.mean(fi.sum(axis=1) / demand.sum(axis=1)))


def capacity_ratio(inputs):
    """Critical ratio for capacity: adding a unit of k costs u every day and
    saves w on overtime days, so the optimum has Pr(P > k) = u/w."""
    return 1 - inputs.u / inputs.w


def _capacity_cost(production, k, u, w):
    return u * k + w * np.maximum(0, production - k)


def _mean_total_cost(ns0, production, ss, k, inputs):
    ns = ns0 + ss
    per_rep = (inputs.h * np.maximum(ns, 0) + inputs.b * np.maximum(-ns, 0)
               + _capacity_cost(production, k, inputs.u, inputs.w)).sum(axis=1)
    return float(per_rep.mean())


# -----------------------------------------------------------------------------
# Single run (what the UI's Run button calls)
# -----------------------------------------------------------------------------

def simulate_run(inputs, c, ss, k, T, seed=42):
    """One replication with the given parameters. Returns the per-day DataFrame
    for the reporting window (day 0 = first day after warm-up)."""
    W = warmup_days(inputs, c)
    N = W + inputs.horizon
    demand, lead = draw_randomness(inputs, N, 1, seed)
    out = _simulate_core(inputs, c, np.array([float(T)]), int(ss), demand, lead)
    s = slice(W, N)

    df = pd.DataFrame({
        "day": np.arange(inputs.horizon),
        "demand": out["demand"][0, s],
        "order": out["order"][0, s],
        "lead_time_of_order": out["lead_time_of_order"][0, s],
        "arrivals": out["arrivals"][0, s],
        "wip": out["wip"][0, s],
        "production": out["production"][0, s],
        "available": out["available"][0, s],
        "fulfilled_immediate": out["fulfilled_immediate"][0, s],
        "fulfilled_backlog": out["fulfilled_backlog"][0, s],
        "net_stock": out["net_stock"][0, s],
    })
    df["inventory"] = df["net_stock"].clip(lower=0)
    df["backlog"] = (-df["net_stock"]).clip(lower=0)
    df["fill_rate_cumulative"] = df["fulfilled_immediate"].cumsum() / df["demand"].cumsum().replace(0, np.nan)
    df["holding_cost"] = inputs.h * df["inventory"]
    df["backlog_cost"] = inputs.b * df["backlog"]
    df["reserved_capacity_cost"] = inputs.u * k
    df["overtime_cost"] = inputs.w * (df["production"] - k).clip(lower=0)
    df["capacity_cost"] = df["reserved_capacity_cost"] + df["overtime_cost"]
    df["total_cost"] = df["holding_cost"] + df["backlog_cost"] + df["capacity_cost"]
    df["cumulative_cost"] = df["total_cost"].cumsum()
    return df


def run_summary(df, c):
    """Headline numbers for a single run DataFrame."""
    bw, nsamp = _block_metrics(df["order"].to_numpy()[None], df["net_stock"].to_numpy()[None],
                               df["demand"].to_numpy()[None], c)
    return {
        "total_cost": float(df["total_cost"].sum()),
        "fill_rate": float(df["fulfilled_immediate"].sum() / df["demand"].sum()),
        "bullwhip": float(bw),
        "nsamp": float(nsamp),
    }


# -----------------------------------------------------------------------------
# Theory
# -----------------------------------------------------------------------------

def theory(inputs, c, T):
    """Closed-form reference values shown in the UI.

    ss: King's formula with z = Phi^-1(target fill rate). King's z is strictly
        a cycle service level, so using the fill rate overstates ss relative to
        the fill-rate-constrained optimum.
    Bullwhip: 1/(2T - 1) for POUT with a mean forecast and equal feedback
        gains (T > 0.5), independent of the lead time.
    k:  mu_d + Phi^-1(1 - u/w) * sigma_P, None when T <= 0.5 (unstable).
        k sits below mean production when w < 2u and above it when w > 2u.
    """
    z_ss = float(norm.ppf(inputs.fill_target))
    R = 0 if c == 1 else c
    ss_theory = z_ss * np.sqrt((inputs.mu_L + R) * inputs.sigma_d ** 2 + (inputs.sigma_L * inputs.mu_d) ** 2)
    z_k = float(norm.ppf(capacity_ratio(inputs)))

    if T > 0.5:
        bullwhip = 1 / (2 * T - 1)
        sigma_q = inputs.sigma_d * np.sqrt(c / (2 * T - 1))    # per order
        sigma_P = sigma_q / c                                  # daily production
        k_theory = inputs.mu_d + z_k * sigma_P
    else:
        bullwhip = sigma_q = sigma_P = k_theory = None

    return {"z_ss": z_ss, "R": R, "ss_theory": float(ss_theory), "bullwhip_theory": bullwhip,
            "sigma_q": sigma_q, "sigma_P": sigma_P, "z_k": z_k, "k_theory": k_theory}


# -----------------------------------------------------------------------------
# Optimisation
# -----------------------------------------------------------------------------
# Orders, production and Bullwhip depend only on T, and NS(ss) = NS(0) + ss.
# So for each T: simulate once with ss = 0, then
#   k*(T)  = (1 - u/w) quantile of pooled daily production
#   ss*(T) = b/(b+h) quantile of pooled -NS(0) (rounded up), raised by integer
#            bisection until the mean fill rate meets the target
# and price (ss*, k*) exactly from the stored arrays.
# -----------------------------------------------------------------------------

def _optimal_ss(ns0, ns_prev0, arrivals, demand, inputs):
    ss = int(np.ceil(np.quantile(-ns0, inputs.b / (inputs.b + inputs.h))))
    if _fill_rate(ns_prev0, arrivals, demand, ss) >= inputs.fill_target:
        return ss
    lo, step = ss, max(1, int(np.ceil(inputs.mu_d)))
    hi = lo + step
    while _fill_rate(ns_prev0, arrivals, demand, hi) < inputs.fill_target:
        lo, step = hi, step * 2
        hi = lo + step
    while hi - lo > 1:                    # lo infeasible, hi feasible
        mid = (lo + hi) // 2
        if _fill_rate(ns_prev0, arrivals, demand, mid) >= inputs.fill_target:
            hi = mid
        else:
            lo = mid
    return hi


def _evaluate_rows(inputs, c, T_values, demand, lead, W):
    """Optimise ss and k for each T in T_values with common random numbers.
    Returns one summary dict per T (no time series kept)."""
    R = demand.shape[0]
    results = []
    for start in range(0, len(T_values), T_CHUNK):
        Ts = np.asarray(T_values[start:start + T_CHUNK], dtype=float)
        out = _simulate_core(inputs, c, np.repeat(Ts, R), 0,
                             np.tile(demand, (len(Ts), 1)), np.tile(lead, (len(Ts), 1)))
        for i, T in enumerate(Ts):
            r = slice(i * R, (i + 1) * R)
            ns0 = out["net_stock"][r, W:]
            ns_prev0 = out["net_stock_prev"][r, W:]
            arr = out["arrivals"][r, W:]
            dem = out["demand"][r, W:]
            prod = out["production"][r, W:]

            k_star = float(np.quantile(prod, capacity_ratio(inputs)))
            ss_star = _optimal_ss(ns0, ns_prev0, arr, dem, inputs)
            bw, nsamp = _block_metrics(out["order"][r, W:], ns0 + ss_star, dem, c)
            results.append({
                "T": float(T),
                "bullwhip": bw,
                "bullwhip_theory": 1 / (2 * T - 1) if T > 0.5 else np.nan,
                "nsamp": nsamp,
                "total_cost": _mean_total_cost(ns0, prod, ss_star, k_star, inputs),
                "ss_star_T": ss_star,
                "k_star_T": k_star,
                "fill_rate": _fill_rate(ns_prev0, arr, dem, ss_star),
            })
    return results


@dataclass
class OptResult:
    ss_star: int
    k_star: float
    T_star: float
    cost: float
    fill_rate: float
    bullwhip: float
    nsamp: float
    sweep: pd.DataFrame = field(repr=False)
    seed: int = 42
    c: int = 1


def optimise(inputs, c, seed=42, reps=N_REPS, T_grid=T_GRID):
    """Minimise expected total cost over the horizon subject to mean fill rate
    >= target, over (ss, k, T) with T restricted to [1, 10]."""
    W = warmup_days(inputs, c)
    demand, lead = draw_randomness(inputs, W + inputs.horizon, reps, seed)

    sweep = pd.DataFrame(_evaluate_rows(inputs, c, list(T_grid), demand, lead, W))
    candidates = sweep[(sweep["T"] >= T_MIN_OPT) & (sweep["T"] <= T_MAX_OPT)
                       & (sweep["fill_rate"] >= inputs.fill_target)]
    best = candidates.loc[candidates["total_cost"].idxmin()].to_dict()

    cache = {}

    def cost_at(T):
        key = round(float(T), 4)
        if key not in cache:
            cache[key] = _evaluate_rows(inputs, c, [key], demand, lead, W)[0]
        return cache[key]["total_cost"]

    lo = max(T_MIN_OPT, best["T"] - REFINE_HALF_WIDTH)
    hi = min(T_MAX_OPT, best["T"] + REFINE_HALF_WIDTH)
    minimize_scalar(cost_at, bounds=(lo, hi), method="bounded", options={"xatol": 0.01})
    for row in cache.values():
        if row["fill_rate"] >= inputs.fill_target and row["total_cost"] < best["total_cost"]:
            best = row

    return OptResult(ss_star=int(best["ss_star_T"]), k_star=float(best["k_star_T"]),
                     T_star=float(best["T"]), cost=float(best["total_cost"]),
                     fill_rate=float(best["fill_rate"]), bullwhip=float(best["bullwhip"]),
                     nsamp=float(best["nsamp"]), sweep=sweep, seed=seed, c=c)


def evaluate_params(inputs, c, ss, k, T, seed=42, reps=N_REPS):
    """Mean cost and fill rate over `reps` replications at fixed (ss, k, T),
    using the same random numbers as optimise()."""
    W = warmup_days(inputs, c)
    demand, lead = draw_randomness(inputs, W + inputs.horizon, reps, seed)
    out = _simulate_core(inputs, c, np.full(reps, float(T)), int(round(ss)), demand, lead)
    s = slice(W, None)
    ns, prod = out["net_stock"][:, s], out["production"][:, s]
    return {
        "total_cost": _mean_total_cost(ns, prod, 0, k, inputs),
        "fill_rate": float(np.mean(out["fulfilled_immediate"][:, s].sum(axis=1) / out["demand"][:, s].sum(axis=1))),
    }
