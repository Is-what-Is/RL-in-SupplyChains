import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict, deque


# =============================================================================
# STAGE 1: CORE SUPPLY CHAIN SIMULATION LOGIC
# =============================================================================


# -----------------------------------------------------------------------------
# Step 1: Demand Generation
# -----------------------------------------------------------------------------
# D_t = max(0, round(mu_D + epsilon_t)),  epsilon_t ~ N(0, sigma_D^2)
# sigma_D = 10 under normal conditions; rises to 40 during demand disruption.
# -----------------------------------------------------------------------------

def generate_demand(mean=50, std=10):
    noise = np.random.normal(0, std)
    return max(0, round(mean + noise))


# -----------------------------------------------------------------------------
# Step 2: POUT Ordering Policy
# -----------------------------------------------------------------------------
# O_t = max(0, round(D_hat + (1/T)(S_t - I_t) + (1/T)*(desired_WIP - actual_WIP)))
# -----------------------------------------------------------------------------

LEAD_TIME  = 5
FEEDBACK_T = 2
DEMAND_MEAN = 50

def compute_order(inventory, safety_stock, order_history):
    demand_forecast      = DEMAND_MEAN
    inventory_correction = (1 / FEEDBACK_T) * (safety_stock - inventory)
    desired_wip          = LEAD_TIME * demand_forecast
    actual_wip           = sum(order_history)
    pipeline_correction  = (1 / FEEDBACK_T) * (desired_wip - actual_wip)
    return max(0, round(demand_forecast + inventory_correction + pipeline_correction))


# -----------------------------------------------------------------------------
# Step 3: Suppliers
# -----------------------------------------------------------------------------
# Q^A_t = alpha * O_t * C^A_t
# Q^B_t = (1 - alpha) * O_t * C^B_t
# -----------------------------------------------------------------------------

ALPHA_OPTIONS = [0.2, 0.5, 0.8]

def compute_supply(order, alpha, capacity_a, capacity_b):
    quantity_a = round(alpha * order * capacity_a)
    quantity_b = round((1 - alpha) * order * capacity_b)
    return quantity_a, quantity_b


# -----------------------------------------------------------------------------
# Step 4: Transport and Routing
# -----------------------------------------------------------------------------
# LT^A = max(1, round(5 + N(0,1) + traffic_a))
# LT^B = max(1, round(5 + N(0,2) + traffic_b))   ← higher variance by design
# -----------------------------------------------------------------------------

ROUTE_A = 0
ROUTE_B = 1
ROUTE_OPTIONS = [ROUTE_A, ROUTE_B]

def compute_lead_time(route, traffic_a=0, traffic_b=0):
    if route == ROUTE_A:
        noise = np.random.normal(0, 1)
        return max(1, round(5 + noise + traffic_a))
    else:
        noise = np.random.normal(0, 2)
        return max(1, round(5 + noise + traffic_b))

def dispatch_order(pipeline, current_step, quantity, lead_time):
    pipeline[current_step + lead_time] += quantity

def receive_delivery(pipeline, current_step):
    return pipeline.pop(current_step, 0)


# -----------------------------------------------------------------------------
# Step 5: Inventory, Backlog, Fulfilment and KPIs
# -----------------------------------------------------------------------------
# Available_t = I_{t-1} + Q_t - B_{t-1}
# I_t  = max(0, Available_t - D_t)
# B_t  = max(0, D_t - Available_t)
# F_t  = max(0, min(Available_t, D_t))
# -----------------------------------------------------------------------------

def update_inventory(prev_inventory, delivered, prev_backlog, demand):
    available  = prev_inventory + delivered - prev_backlog
    inventory  = max(0, available - demand)
    backlog    = max(0, demand - available)
    fulfilment = max(0, min(available, demand))
    return inventory, backlog, available, fulfilment

def compute_fill_rate(fulfilment, demand):
    if demand == 0:
        return 1.0
    return fulfilment / demand

def compute_nsamp(inventory_history, demand_history):
    if len(inventory_history) < 2:
        return 0.0
    var_demand = np.var(demand_history)
    if var_demand == 0:
        return 0.0
    return np.var(inventory_history) / var_demand


# -----------------------------------------------------------------------------
# Step 6: Disruption Scenarios
# -----------------------------------------------------------------------------
# Nine scenarios: scenario 0 is a no-disruption baseline; scenarios 1-7 cover
# all single and pairwise disruption combinations and are used during training
# (randomly sampled each episode); scenario 8 (all three disruptions) is held
# out for generalisation testing.
#
# For each scenario:
#   disruptions — which disruption types are active (empty list = no disruption)
#   duration    — how many steps each individual disruption lasts
#   max_start   — latest step a disruption can begin (ensures it ends by 365)
#
# Scenario 0 has no disruptions so duration and max_start are unused.
# For scenarios 1-4 each episode draws one random start uniformly from
# [1, max_start].  For scenarios 5-8 each disruption type draws its own
# independent random start, so the windows may overlap, be contiguous, or
# be entirely separate — giving the agent a wider variety of states.
# -----------------------------------------------------------------------------

SAFETY_STOCK_OPTIONS = [20, 29, 115]
N_STEPS = 365

SCENARIO_CONFIGS = {
    0: {"disruptions": [],                                        "duration": 0,  "max_start": 365},
    1: {"disruptions": ["supplier_a"],                            "duration": 90, "max_start": 275},
    2: {"disruptions": ["supplier_b"],                            "duration": 90, "max_start": 275},
    3: {"disruptions": ["route_a"],                               "duration": 90, "max_start": 275},
    4: {"disruptions": ["demand"],                                "duration": 90, "max_start": 275},
    5: {"disruptions": ["supplier_a", "route_a"],                 "duration": 90, "max_start": 275},
    6: {"disruptions": ["supplier_a", "demand"],                  "duration": 90, "max_start": 275},
    7: {"disruptions": ["route_a",    "demand"],                  "duration": 90, "max_start": 275},
    8: {"disruptions": ["supplier_a", "route_a", "demand"],       "duration": 90, "max_start": 275},
}


def get_disruption_params(step, disruption_windows):
    """
    Return (capacity_a, capacity_b, traffic_a, traffic_b, demand_std) for
    the current step.

    disruption_windows: dict mapping disruption name -> (start, end).
    Each disruption type is checked independently, so multiple disruptions
    can have non-overlapping windows within the same episode.
    """
    def _active(name):
        if name not in disruption_windows:
            return False
        s, e = disruption_windows[name]
        return s <= step <= e

    capacity_a = 0.5 if _active("supplier_a") else 1.0
    capacity_b = 0.5 if _active("supplier_b") else 1.0
    traffic_a  = 5   if _active("route_a")    else 0
    demand_std = 40  if _active("demand")     else 10
    return capacity_a, capacity_b, traffic_a, 0, demand_std


def run_episode(policy_fn, seed=None, scenario=None, disruption_starts=None):
    """
    Run a single 365-step episode using policy_fn(obs) -> (alpha, route, safety_stock).

    scenario          : integer 0-8, or None to sample randomly from 0-7.
    disruption_starts : list of ints (one per disruption in the scenario), or None to
                        sample each independently within the scenario's valid range.

    Returns a list of per-step record dicts including all KPIs and disruption info.
    """
    if seed is not None:
        np.random.seed(seed)

    if scenario is None:
        scenario = int(np.random.randint(0, 8))   # 0 to 7 inclusive

    config = SCENARIO_CONFIGS[scenario]

    if disruption_starts is None:
        disruption_starts = [
            int(np.random.randint(1, config["max_start"] + 1))
            for _ in config["disruptions"]
        ]

    disruption_windows = {
        name: (start, start + config["duration"] - 1)
        for name, start in zip(config["disruptions"], disruption_starts)
    }

    # initialise state — pipeline pre-load prevents cold-start transient
    inventory     = SAFETY_STOCK_OPTIONS[1]   # start at 29 (middle safety stock)
    backlog       = 0
    pipeline      = defaultdict(int)
    for i in range(1, LEAD_TIME + 1):
        pipeline[i] = DEMAND_MEAN
    order_history     = deque([DEMAND_MEAN] * LEAD_TIME, maxlen=LEAD_TIME)
    inventory_history = []
    demand_history    = []
    records           = []

    for step in range(1, N_STEPS + 1):

        # 1. disruption conditions for this step
        cap_a, cap_b, traf_a, traf_b, dem_std = get_disruption_params(
            step, disruption_windows
        )

        # 2. receive deliveries
        delivered = receive_delivery(pipeline, step)

        # 3. generate demand
        demand = generate_demand(std=dem_std)

        # 4. build observation and query policy
        nsamp     = compute_nsamp(inventory_history, demand_history)
        fill_rate = compute_fill_rate(
            max(0, min(inventory + delivered - backlog, demand)), demand
        )
        obs = [step, inventory, backlog, inventory + delivered - backlog,
               demand, 0, cap_a, cap_b, traf_a, traf_b, fill_rate, nsamp, dem_std]

        alpha, route, safety_stock = policy_fn(obs)

        # 5. compute order and supplier split
        order          = compute_order(inventory, safety_stock, order_history)
        qty_a, qty_b   = compute_supply(order, alpha, cap_a, cap_b)
        total_supplied = qty_a + qty_b

        # 6. dispatch into pipeline
        lead_time = compute_lead_time(route, traf_a, traf_b)
        dispatch_order(pipeline, step, total_supplied, lead_time)
        order_history.append(order)

        # 7. update inventory accounting
        inventory, backlog, available, fulfilment = update_inventory(
            inventory, delivered, backlog, demand
        )

        # 8. compute KPIs
        inventory_history.append(inventory)
        demand_history.append(demand)
        fill_rate = compute_fill_rate(fulfilment, demand)
        nsamp     = compute_nsamp(inventory_history, demand_history)

        records.append({
            "step":             step,
            "inventory":        inventory,
            "backlog":          backlog,
            "available":        available,
            "demand":           demand,
            "order":            order,
            "delivered":        delivered,
            "capacity_a":       cap_a,
            "capacity_b":       cap_b,
            "traffic_a":        traf_a,
            "traffic_b":        traf_b,
            "demand_std":       dem_std,
            "alpha":            alpha,
            "route":            route,
            "safety_stock":     safety_stock,
            "fill_rate":        fill_rate,
            "nsamp":            nsamp,
            "scenario":           scenario,
            "disruption_windows": disruption_windows,
        })

    return records


# =============================================================================
# STAGE 2: GYMNASIUM ENVIRONMENT WRAPPER
# =============================================================================
# Action space: Discrete(18) — 3 alpha x 2 routes x 3 safety stocks
# Observation space: Box(13,) float32
#   [step, inventory, backlog, available, demand, order,
#    capacity_a, capacity_b, traffic_a, traffic_b,
#    fill_rate, nsamp, demand_std]
# Reward: R_t = 100*fill_rate - 2*nsamp - 0.5*backlog - 0.1*inventory
# =============================================================================

class SupplyChainEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, scenario=None, disruption_starts=None):
        """
        scenario          : fix a scenario for every episode (None = random 1-7).
        disruption_starts : list of ints fixing the start step for each disruption
                            in the scenario (None = sample randomly each reset).

        Pass fixed values here when running comparison episodes so that all
        policies face identical conditions.
        """
        super().__init__()
        self._fixed_scenario          = scenario
        self._fixed_disruption_starts = disruption_starts

        # ------------------------------------------------------------------
        # Action space: 3 alpha × 2 routes × 3 safety stocks = 18 actions
        # ------------------------------------------------------------------
        self.action_map = [
            (alpha, route, ss)
            for alpha in ALPHA_OPTIONS
            for route  in ROUTE_OPTIONS
            for ss     in SAFETY_STOCK_OPTIONS
        ]
        self.action_space = spaces.Discrete(len(self.action_map))

        # ------------------------------------------------------------------
        # Observation space: 13 elements (added demand_std at index 12)
        # ------------------------------------------------------------------
        low  = np.array([1, 0, 0, -2000, 0, 0, 0, 0, 0, 0, 0, 0,    10], dtype=np.float32)
        high = np.array([N_STEPS, 2000, 2000, 2000, 300, 500, 1, 1, 10, 10, 1, 50000, 40], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    # Samples (or uses fixed) scenario and disruption start, then
    # reinitialises all simulation state.
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        scenario          = self._fixed_scenario
        disruption_starts = self._fixed_disruption_starts

        if scenario is None:
            scenario = int(np.random.randint(0, 8))   # 0-7 for training

        config = SCENARIO_CONFIGS[scenario]

        if disruption_starts is None:
            disruption_starts = [
                int(np.random.randint(1, config["max_start"] + 1))
                for _ in config["disruptions"]
            ]

        self._scenario           = scenario
        self._disruption_windows = {
            name: (start, start + config["duration"] - 1)
            for name, start in zip(config["disruptions"], disruption_starts)
        }

        # simulation state
        self._step              = 1
        self._inventory         = SAFETY_STOCK_OPTIONS[1]   # 30
        self._backlog           = 0
        self._available         = SAFETY_STOCK_OPTIONS[1]
        self._pipeline          = defaultdict(int)
        for i in range(1, LEAD_TIME + 1):
            self._pipeline[i] = DEMAND_MEAN
        self._order_history     = deque([DEMAND_MEAN] * LEAD_TIME, maxlen=LEAD_TIME)
        self._inventory_history = []
        self._demand_history    = []

        # initialise disruption params so the first observation is meaningful
        (self._cap_a, self._cap_b,
         self._traf_a, self._traf_b,
         self._dem_std) = get_disruption_params(1, self._disruption_windows)
        self._demand    = 0.0
        self._order     = 0.0
        self._fill_rate = 1.0
        self._nsamp     = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    def step(self, action):
        alpha, route, safety_stock = self.action_map[action]

        # 1. disruption conditions for this step
        (self._cap_a, self._cap_b,
         self._traf_a, self._traf_b,
         self._dem_std) = get_disruption_params(self._step, self._disruption_windows)

        # 2. receive deliveries
        delivered = receive_delivery(self._pipeline, self._step)

        # 3. generate demand
        self._demand = generate_demand(std=self._dem_std)

        # 4. compute order and supplier split
        self._order    = compute_order(self._inventory, safety_stock, self._order_history)
        qty_a, qty_b   = compute_supply(self._order, alpha, self._cap_a, self._cap_b)
        total_supplied = qty_a + qty_b

        # 5. dispatch into pipeline
        lead_time = compute_lead_time(route, self._traf_a, self._traf_b)
        dispatch_order(self._pipeline, self._step, total_supplied, lead_time)
        self._order_history.append(self._order)

        # 6. update inventory accounting
        self._inventory, self._backlog, self._available, fulfilment = update_inventory(
            self._inventory, delivered, self._backlog, self._demand
        )

        # 7. compute KPIs
        self._inventory_history.append(self._inventory)
        self._demand_history.append(self._demand)
        self._fill_rate = compute_fill_rate(fulfilment, self._demand)
        self._nsamp     = compute_nsamp(self._inventory_history, self._demand_history)

        # 8. reward: R_t = 100*fill_rate - nsamp - 0.5*backlog - 0.1*inventory
        reward = (100 * self._fill_rate
                  - 2 * self._nsamp
                  - 0.5 * self._backlog
                  - 0.1 * self._inventory)

        self._step += 1
        terminated = self._step > N_STEPS

        info = {
            "step":             self._step - 1,
            "inventory":        self._inventory,
            "backlog":          self._backlog,
            "available":        self._available,
            "demand":           self._demand,
            "order":            self._order,
            "delivered":        delivered,
            "capacity_a":       self._cap_a,
            "capacity_b":       self._cap_b,
            "traffic_a":        self._traf_a,
            "traffic_b":        self._traf_b,
            "demand_std":       self._dem_std,
            "fill_rate":        self._fill_rate,
            "nsamp":            self._nsamp,
            "alpha":            alpha,
            "route":            route,
            "safety_stock":     safety_stock,
            "scenario":           self._scenario,
            "disruption_windows": self._disruption_windows,
        }

        return self._get_obs(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # _get_obs()
    # ------------------------------------------------------------------
    # 13-element observation vector. demand_std (index 12) makes the demand
    # disruption directly observable so the agent and heuristic can respond.
    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.array([
            self._step,
            self._inventory,
            self._backlog,
            self._available,
            self._demand,
            self._order,
            self._cap_a,
            self._cap_b,
            self._traf_a,
            self._traf_b,
            self._fill_rate,
            self._nsamp,
            self._dem_std,
        ], dtype=np.float32)
