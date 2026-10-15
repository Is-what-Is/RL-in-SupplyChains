"""
evaluate_dqn.py  —  Load the saved DQN agent and compare against baselines.

Loads best_model.zip and vec_normalize.pkl from dqn_output/ without
retraining. Runs one deterministic episode (e.g. Scenario 5: Supplier A + Route A
disruption) for the DQN and three baseline policies, then prints a summary
table and saves comparison plots.

Usage:
    python evaluate_dqn.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from supply_chain_sim import (
    SupplyChainEnv,
    run_episode,
    ALPHA_OPTIONS,
    ROUTE_OPTIONS,
    SAFETY_STOCK_OPTIONS,
    SCENARIO_CONFIGS,
)
from partial_obs_wrapper import PartialObsWrapper

SAVE_DIR = "dqn_output"
SCENARIO = 0
SEED     = 42

# Pre-determine fixed disruption starts (one per disruption type) so every
# policy faces identical conditions.
np.random.seed(SEED)
_cfg               = SCENARIO_CONFIGS[SCENARIO]
DISRUPTION_STARTS  = [
    int(np.random.randint(1, _cfg["max_start"] + 1))
    for _ in _cfg["disruptions"]
]

print(f"Scenario : {SCENARIO}  ({', '.join(_cfg['disruptions'])})")
for _name, _start in zip(_cfg["disruptions"], DISRUPTION_STARTS):
    print(f"  {_name}: steps {_start} – {_start + _cfg['duration'] - 1}")
print()


# =============================================================================
# LOAD SAVED MODEL
# =============================================================================
def make_env():
    return PartialObsWrapper(SupplyChainEnv())

best_model = DQN.load(f"{SAVE_DIR}/best_model")

test_env = make_vec_env(make_env, n_envs=1, seed=SEED)
test_env = VecNormalize.load(f"{SAVE_DIR}/vec_normalize.pkl", test_env)
test_env.training    = False
test_env.norm_reward = False

mean_reward, std_reward = evaluate_policy(
    best_model, test_env, n_eval_episodes=20, deterministic=True
)
print(f"Test (20 episodes): mean episode reward = {mean_reward:.1f} ± {std_reward:.1f}\n")


# =============================================================================
# RUN ONE DQN EPISODE
# =============================================================================
def run_dqn_episode(model, scenario, disruption_starts, seed=SEED):
    """Run a single deterministic episode and return a list of info dicts."""
    env = make_vec_env(
        lambda: PartialObsWrapper(SupplyChainEnv(scenario=scenario, disruption_starts=disruption_starts)),
        n_envs=1, seed=seed,
    )
    env = VecNormalize.load(f"{SAVE_DIR}/vec_normalize.pkl", env)
    env.training    = False
    env.norm_reward = False

    obs     = env.reset()
    records = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        records.append(info[0])
        if done[0]:
            break
    env.close()
    return records


dqn_records = run_dqn_episode(
    best_model,
    scenario          = SCENARIO,
    disruption_starts = DISRUPTION_STARTS,
)


# =============================================================================
# BASELINE EPISODES
# obs indices (13 elements):
#   0=step, 1=inventory, 2=backlog, 3=available, 4=demand, 5=order,
#   6=capacity_a, 7=capacity_b, 8=traffic_a, 9=traffic_b,
#   10=fill_rate, 11=nsamp, 12=demand_std
# =============================================================================
def fixed_policy(obs):
    return 0.5, 0, 20

_rng = np.random.default_rng(99)

def random_policy(obs):
    return (
        _rng.choice(ALPHA_OPTIONS),
        int(_rng.choice(ROUTE_OPTIONS)),
        int(_rng.choice(SAFETY_STOCK_OPTIONS)),
    )

def rule_based_policy(obs):
    _, _, _, _, _, _, cap_a, cap_b, traf_a, _, _, _, demand_std = obs

    if cap_a < 1.0:
        alpha = 0.2
    elif cap_b < 1.0:
        alpha = 0.8
    else:
        alpha = 0.5

    route = 1 if traf_a > 0 else 0

    if demand_std > 10:
        ss = 115
    elif demand_std == 10:
        ss = 28
    else:
        ss = 20

    return alpha, route, ss

fixed_records  = run_episode(fixed_policy,      seed=SEED,
                             scenario=SCENARIO, disruption_starts=DISRUPTION_STARTS)
random_records = run_episode(random_policy,     seed=SEED,
                             scenario=SCENARIO, disruption_starts=DISRUPTION_STARTS)
rule_records   = run_episode(rule_based_policy, seed=SEED,
                             scenario=SCENARIO, disruption_starts=DISRUPTION_STARTS)


# =============================================================================
# EXTRACT METRICS
# =============================================================================
def extract(records):
    steps      = [r.get("step", i + 1) for i, r in enumerate(records)]
    fill_rate  = [r["fill_rate"]  for r in records]
    nsamp      = [r["nsamp"]      for r in records]
    inventory  = [r["inventory"]  for r in records]
    backlog    = [r["backlog"]    for r in records]
    reward     = [100*r["fill_rate"] - 2*r["nsamp"] - 0.5*r["backlog"] - 0.1*r["inventory"]
                  for r in records]
    mean_reward = np.mean(reward)
    cum_reward  = list(np.cumsum(reward))
    alpha       = [r["alpha"]         for r in records]
    route       = [r["route"]         for r in records]
    safety_stock = [r["safety_stock"] for r in records]
    return (steps, fill_rate, nsamp, inventory, backlog, reward, mean_reward, cum_reward,
            alpha, route, safety_stock)


f_s, f_fr, f_ns, f_i, f_b, f_r, f_mr, f_cr, f_al, f_rt, f_ss = extract(fixed_records)
r_s, r_fr, r_ns, r_i, r_b, r_r, r_mr, r_cr, r_al, r_rt, r_ss = extract(random_records)
b_s, b_fr, b_ns, b_i, b_b, b_r, b_mr, b_cr, b_al, b_rt, b_ss = extract(rule_records)
d_s, d_fr, d_ns, d_i, d_b, d_r, d_mr, d_cr, d_al, d_rt, d_ss = extract(dqn_records)


# =============================================================================
# PRINT RESULTS TABLE
# =============================================================================
print(f"Policy comparison — Scenario {SCENARIO}: {', '.join(_cfg['disruptions'])}")
for _name, _start in zip(_cfg["disruptions"], DISRUPTION_STARTS):
    print(f"  {_name}: steps {_start} – {_start + _cfg['duration'] - 1}")
print()
print(f"  {'Policy':<14} {'Avg Fill Rate':>14} {'Avg NSAmp':>10} "
      f"{'Avg Inventory':>14} {'Avg Backlog':>11} {'Avg Reward':>11} {'Total Reward':>13}")
print("  " + "-" * 89)
for label, fr, ns, inv, bl, mr, cr in [
    ("Fixed",      f_fr, f_ns, f_i, f_b, f_mr, f_cr),
    ("Random",     r_fr, r_ns, r_i, r_b, r_mr, r_cr),
    ("Rule-based", b_fr, b_ns, b_i, b_b, b_mr, b_cr),
    ("DQN",        d_fr, d_ns, d_i, d_b, d_mr, d_cr),
]:
    print(f"  {label:<14} {np.mean(fr):>14.3f} {np.mean(ns):>10.2f} "
          f"{np.mean(inv):>14.1f} {np.mean(bl):>11.1f} {mr:>11.2f} {cr[-1]:>13.1f}")


# =============================================================================
# DISRUPTION SHADING — dynamic, derived from episode records
# =============================================================================
DISRUPTION_STYLES = {
    "supplier_a": ("#e74c3c", "Supplier A disruption"),
    "supplier_b": ("#c0392b", "Supplier B disruption"),
    "route_a":    ("#e67e22", "Route A disruption"),
    "demand":     ("#9b59b6", "Demand disruption"),
}


def shade_disruptions(ax, records):
    d_windows = records[0].get("disruption_windows", {})
    for name, (start, end) in d_windows.items():
        if name in DISRUPTION_STYLES:
            ax.axvspan(start, end, alpha=0.15, color=DISRUPTION_STYLES[name][0], zorder=0)


def get_disruption_patches(records):
    d_windows = records[0].get("disruption_windows", {})
    return [
        mpatches.Patch(color=DISRUPTION_STYLES[name][0], alpha=0.4,
                       label=DISRUPTION_STYLES[name][1])
        for name in d_windows
        if name in DISRUPTION_STYLES
    ]


# =============================================================================
# POLICY COMPARISON PLOT
# =============================================================================
COLOURS = {
    "Fixed":      "#2980b9",
    "Random":     "#e67e22",
    "Rule-based": "#27ae60",
    "DQN":        "#8e44ad",
}

metrics = [
    ("Fill Rate",         f_fr, r_fr, b_fr, d_fr, "Rate [0–1]"),
    ("NSAmp",             f_ns, r_ns, b_ns, d_ns, "Ratio"),
    ("Inventory",         f_i,  r_i,  b_i,  d_i,  "Units"),
    ("Backlog",           f_b,  r_b,  b_b,  d_b,  "Units"),
    ("Reward",            f_r,  r_r,  b_r,  d_r,  "R_t"),
    ("Cumulative Reward", f_cr, r_cr, b_cr, d_cr,  "Cumulative R"),
]

fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)), sharex=True)
fig.suptitle(
    f"Supply Chain — Policy Comparison  |  Scenario {SCENARIO}: "
    f"{', '.join(_cfg['disruptions'])}",
    fontsize=13, fontweight="bold",
)

for ax, (title, fv, rv, bv, dv, ylabel) in zip(axes, metrics):
    shade_disruptions(ax, fixed_records)
    ax.plot(f_s, fv, label="Fixed",      color=COLOURS["Fixed"],      lw=1.2, alpha=0.85)
    ax.plot(r_s, rv, label="Random",     color=COLOURS["Random"],     lw=1.2, alpha=0.85)
    ax.plot(b_s, bv, label="Rule-based", color=COLOURS["Rule-based"], lw=1.5)
    ax.plot(d_s, dv, label="DQN",        color=COLOURS["DQN"],        lw=1.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)

# Policy legend already on each axis; add disruption patches to the last axis.
disruption_patches = get_disruption_patches(fixed_records)
if disruption_patches:
    axes[-1].add_artist(axes[-1].get_legend())
    axes[-1].legend(handles=disruption_patches, fontsize=8, loc="lower left")

axes[-1].set_xlabel("Simulation Step (Day)", fontsize=10)

plt.tight_layout()
out_path = f"{SAVE_DIR}/policy_comparison_dqn_s{SCENARIO}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {out_path}")


# =============================================================================
# ACTIONS PLOT — alpha / route / safety stock chosen by each policy over time
# =============================================================================
action_metrics = [
    ("Alpha (supplier split)", b_al, d_al, "Alpha"),
    ("Route",                  b_rt, d_rt, "Route"),
    ("Safety Stock",           b_ss, d_ss, "Units"),
]

fig, axes = plt.subplots(len(action_metrics), 1, figsize=(14, 4 * len(action_metrics)), sharex=True)
fig.suptitle(
    f"Supply Chain — Policy Actions  |  Scenario {SCENARIO}: "
    f"{', '.join(_cfg['disruptions'])}",
    fontsize=13, fontweight="bold",
)

for ax, (title, bv, dv, ylabel) in zip(axes, action_metrics):
    shade_disruptions(ax, fixed_records)
    ax.step(b_s, bv, label="Rule-based", color=COLOURS["Rule-based"], lw=1.5, where="post")
    ax.step(d_s, dv, label="DQN",        color=COLOURS["DQN"],        lw=1.5, where="post")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)

axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Route A", "Route B"])

disruption_patches = get_disruption_patches(fixed_records)
if disruption_patches:
    axes[-1].add_artist(axes[-1].get_legend())
    axes[-1].legend(handles=disruption_patches, fontsize=8, loc="lower left")

axes[-1].set_xlabel("Simulation Step (Day)", fontsize=10)

plt.tight_layout()
actions_path = f"{SAVE_DIR}/policy_actions_dqn_s{SCENARIO}.png"
fig.savefig(actions_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {actions_path}")
