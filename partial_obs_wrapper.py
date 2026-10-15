"""
partial_obs_wrapper.py

Gymnasium ObservationWrapper that strips the three directly-observable
disruption parameters from the agent's view:

    Removed:
        index 6  — capacity_a   (agent cannot see supplier capacity)
        index 7  — capacity_b
        index 12 — demand_std   (agent cannot see the true noise parameter)

    Kept (10 elements):
        index 0  — step
        index 1  — inventory
        index 2  — backlog
        index 3  — available
        index 4  — demand
        index 5  — order
        index 8  — traffic_a    (kept: route congestion is plausibly observable)
        index 9  — traffic_b
        index 10 — fill_rate
        index 11 — nsamp

Rationale
---------
In a real supply chain an agent cannot directly observe a supplier's
capacity multiplier or the true demand noise parameter — it can only see
the consequences:
    • Fewer deliveries → rising backlog, falling fill rate  (supplier disruption)
    • Higher / more variable observed demand, rising nsamp  (demand spike)

Keeping these signals hidden forces the DQN to learn disruption detection
from observable consequences rather than from the disruption flag itself,
which should produce a more generalisable policy.

The underlying SupplyChainEnv, its info dict, and run_episode are all
unchanged. The full 13-element observation is still used by the rule-based
baseline policy (which represents a "perfect information" upper bound).

Usage
-----
When you are ready to train the partial-obs agent, replace make_env() in
train_dqn.py with:

    from partial_obs_wrapper import PartialObsWrapper

    def make_env():
        return PartialObsWrapper(SupplyChainEnv())

Everything else in train_dqn.py stays identical. VecNormalize will
automatically operate on the 10-element observation space.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PartialObsWrapper(gym.ObservationWrapper):
    """
    Strips capacity_a, capacity_b and demand_std from the observation so the
    DQN must infer disruptions from observable consequences.

    Full obs  (13): step, inv, backlog, avail, demand, order,
                    capacity_a, capacity_b, traffic_a, traffic_b,
                    fill_rate, nsamp, demand_std

    Partial obs (10): step, inv, backlog, avail, demand, order,
                      traffic_a, traffic_b, fill_rate, nsamp
    """

    KEEP = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11])

    def __init__(self, env):
        super().__init__(env)
        low  = env.observation_space.low[self.KEEP]
        high = env.observation_space.high[self.KEEP]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[self.KEEP]
