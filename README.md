# Reinforcement Learning Supply Chain

This project was undertaken in partial fulfillment of the requirements for the degree of Master of Science (MSc) in Data Science at the University of Exeter.

## Overview
This project implements a simplified supply chain simulation using Gymnasium to investigate whether a Reinforcement Learning (RL) agent can learn effective disruption recovery policies. 

The simulation represents a single-echelon inventory system supplied by two identical but alternative suppliers with two alternative distribution routes. 

The environment is designed as a proof-of-concept digital twin that is able to capture all the simulated data and the RL agent is a proof-of-concept for autonomous decision support. 

The RL agent will be exposed to simulated supply chain disruptions and expected to learn optimal recovery polcies via a Double Deep Q-Network. 

The primary objective is to learn adaptive inventory and sourcing policies that maximise service level while minimising inventory, backlog, and bullwhip effects.

## Installation

Requires Python 3.10+ (Stable-Baselines3 needs Python >= 3.9 and PyTorch >= 2.8, which it will install as a dependency).

```bash
pip install numpy gymnasium matplotlib "stable-baselines3>=2.0" rich
```

| Package | Used for |
|---|---|
| `numpy` | array/vector operations throughout the simulation |
| `gymnasium` | the RL environment interface that `SupplyChainEnv` implements |
| `matplotlib` | learning curve and policy comparison plots |
| `stable-baselines3` | the DQN algorithm (installs PyTorch as a dependency) |
| `rich` | the training progress bar (`model.learn(progress_bar=True)`) |

## Usage

The project files, in the order they depend on each other:

* [`supply_chain_sim.py`](supply_chain_sim.py) — the core simulation and `SupplyChainEnv` Gymnasium environment. Not run directly; imported by the training/evaluation scripts.
* [`partial_obs_wrapper.py`](partial_obs_wrapper.py) — `PartialObsWrapper`, which hides supplier capacity and true demand noise from the agent's observation. Also imported, not run directly.
* [`train_dqn.py`](train_dqn.py) — trains the DQN agent and writes checkpoints, normalisation stats, and a learning curve plot to `dqn_output/`.
* [`evaluate_dqn.py`](evaluate_dqn.py) — loads the trained model from `dqn_output/` (no retraining) and compares it against Fixed, Random, and Rule-based baseline policies on a chosen disruption scenario.

**1. Train the agent:**

```bash
python train_dqn.py
```

This runs the full `TOTAL_TIMESTEPS` (912,500 steps, ~2,500 episodes) training loop and can take a while on CPU. To do a quicker test run, reduce `TOTAL_TIMESTEPS` and `EVAL_FREQ` near the top of the file. On completion it writes to `dqn_output/`:

* `best_model.zip` — checkpoint with the highest evaluation reward
* `dqn_final.zip` — model weights at the end of training
* `vec_normalize.pkl` — observation/reward normalisation statistics (needed to reload the model correctly)
* `evaluations.npz` — per-checkpoint evaluation rewards
* `learning_curve.png` — DQN training progress plotted against Fixed and Rule-based baselines

**2. Evaluate against baselines:**

```bash
python evaluate_dqn.py
```

This loads `best_model.zip` and `vec_normalize.pkl` from `dqn_output/` and runs one episode per policy (Fixed, Random, Rule-based, DQN) on the scenario set by the `SCENARIO` constant near the top of the file (0 = no disruption, 1–7 = training scenarios, 8 = held-out generalisation test). To evaluate a different scenario, change `SCENARIO` and rerun. It saves, per scenario `N`:

* `dqn_output/policy_comparison_dqn_sN.png` — fill rate, NSAmp, backlog, inventory reward, and cumulative reward over time for all four policies
* `dqn_output/policy_actions_dqn_sN.png` — the alpha/route/safety-stock actions chosen by the Rule-based and DQN policies over time

## Supply Chain
To simulate a supply chain we have to define a set of functions to model it...

### Demand 
Demand is generated stochastically, with a mean and random noise:
```math
D_t =  max(0,round(\mu_D + \epsilon_t)) \\
\epsilon_t \sim N(0,\sigma^2_D)
```
Where:

* $D_t$ is the demand at time t
* $\mu_D$ is the mean demand
* $\epsilon_t$ is random demand noise

### Orders
Literature suggests following a Proportional-OUT (POUT) policy is generally optimal over long lead times. Given this single-echelon supply chain does **not** include manufacturing the POUT policy simplifies to:
```math
O_t = max(0,round(\hat{D_{t+L+1}} + \frac{1}{T}(S_t - I_t) + \frac{1}{T}\left(\sum_{i=t+1}^{L}(\hat{D_{t,t+i}} - O_{t-i})\right)))
```
Where:

* $\hat{D_{t+L+1}}$ is the forecast of demand at time t+L+1, in this case it is equal to $\mu_D$
* $L$ is the lead time, in this supply chain it is just the transport time assumed to be 5 'business' days
* $S_t$ is the safety or target stock to hold at time t
* $I_t$ is the inventory held at time t
* The sum of demand over the lead-time $\sum_{i=t+1}^{L}(\hat{D_{t,t+i}})$ is the desired work-in-progress, in this case it is equal to $L*\mu_D$
* $\sum_{i=t+1}^{L}(O_{t-i})$ are all the previous orders placed during the lead time but not recieved.
* $T$ is the proportional feedback controller, typically $0\le1/T\le2$. I will set $T=2$.

### Suppliers
For simplicity, suppliers are assumed to be identical and supply finished goods or orders placed by the wholesaler in a closed system where under no disruption all orders can be satisfied. 
```math
Q^A_t = \alpha_t * O_t *C^A_t \\
Q^B_t = (1-\alpha_t) * O_t *C^B_t
```
Where:

* $Q^A_t$ is the quantity supplied by supplier A at time t
* $Q^B_t$ is the quantity supplied by supplier B at time t
* $\alpha_t$ represents the supplier split at time t being supplied from supplier A
* $C^A_t \in [0,1] $ is the capcity of supplier A
* $C^B_t \in [0,1]$ is the capacity of supplier B

### Distribution
Both suppliers deliver supply via route A or route B. Both routes have an expected delivery time of 5 days. However route B has is more uncertain and has more noise.
```math
LT^A_t = max(1, round(5+\eta^A_t+Traffic^A_t))\\
LT^B_t = max(1, round(5+\eta^B_t+Traffic^B_t))
```
Where:

* $LT^A_t$ is the transportation lead time for route A at time t
* $LT^B_t$ is the transportation lead time for route B at time t
* $\eta^A_t\sim N(0,1)$ is the random noise on route A 
* $\eta^B_t\sim N(0,2)$ is the random noise on route B. Notice route B has more moise than route A. This is by design to encourage route A under normal conditions.
* $Traffic^A_t \geq 0$ is the traffic disruption parameter for route A
* $Traffic^B_t \geq 0$ is the traffic disruption parameter for route B

### Warehouse Inventory and Fulfilment
Total ordered quantities start arriving into the warehouse as inventory and used satisfy backlog demands and current demand.

**Total delivered quantity:**
```math
Delivered_t = Q^A_{t-L} + Q^B_{t-L}
```

**Available Inventory:**
```math
Available_t = I_{t-1} + Delivered_t -B_{t-1}
```
Where:

* $I_{t-1}$ is the inventory remaining from the previous period
* $Q_t$ is the delivered supply at time t
* $B_{t-1}$ is the outstanding backlog from the previous period i.e. the remaining demand that was not satisfied.

**Inventory:**
```math
I_t = max(0,Available_t-D_t)
```

**Backlog:**
```math
B_t = max(0,D_t-Available_t)
```

**Fulfilment:**

This function is used to generate a key performance indicator (KPI). It signifies whether the aivailable inventory at time t is able to meet demand at time t.
```math
F_t = max(0, min(Available_t,D_t))
```
### KPIs

**Fill Rate:**

The fill rate represents the proportion of demand fulfilled directly from inventory. A perfect fill rate would be equal to 1, where in every period $Available_t \ge D_t$
```math
FillRate_t = \frac{F_t}{D_t}
```
For this supply chain I wish to achieve a fill rate of 90% under normal conditions.


**Net Stock Amplification (NSAmp) Ratio**

The NSAmp ratio measures the variability of your inventory as a proportion of the variability of demand. In this supply chain setting we are more concerened about warehouse and inventory management. Therefore, the NSAmp ratio is an important supply chain KPI we want to minimise...
```math
NSAmp_t = \frac{Var(I_t)}{Var(D_t)}
```
Where:

* $NSAmp_t \ge 1 + L$ i.e. it is theoretically impossible for the NSAmp ration to be less than one plus the lead time.

**Bullwhip:**

The bullwhip effect is a well known phenomenon is supply chains and something that you want to minimise. It measures how much your orders vary in proportion to demand variance. This is a important metric when concerned with the load or stress you put on production. The main goal of minimising order variance in resposne to demand variance is to prevent producing lots of things one day and nothing the next, and creating a steady production cycle.
```math
Bullwhip_t = \frac{Var(O_t)}{Var(D_t)}
```
However, because we are not concerned with production in this simplified supply chain it will not be included in the reward function and serves as a reference for consideration.

### Time Steps

This supply chain will run on daily time steps for 365 steps, representing a whole year. 

### Disruption Scenarios

**Supplier Disruption**

This disruption is supposed to represent an unexpected supplier failure resulting in reduced capacity:

$$C^A_t = 0.5 \text{ or } C^B_t = 0.5$$

The optimal response will be to increase $\alpha_t$ if supplier B experiences a capacity disruption, and decrease $\alpha_t$ if supplier A experiences a capacity disruption.


**Route Disruption**

Under normal conditions the traffic parameters for route A and B will be equal to zero. A traffic disruption will increase the delivery lead time for route A:

$$Traffic^A_t = 5$$


**Demand Disruption**

This is an increase in the demand noise variance to simulate volatile demand:

$$\epsilon_t \sim(0,40)$$



**Scenario Implementation**

The supply chain simulation has the option to run the following scenario combinations:

0. No disruption. 

1. Supplier A capacity disruption 

2. Supplier B capacity disruption

3. Route A traffic disruption 

4. Demand disruption 

5. Supplier A capacity disruption and route A traffic disruption

6. Supplier A capacity disruption and demand disruption

7. Route A traffic disruption and demand disruption

8. All three disruption types occur randomly

All disruptions can occur randomly between time step 0 and 365. However, to configure the supply chain disruption scenarios see `supply_chain_sim.py`.

The supply chain step function will take a input disruption parameter which by default is a random number generator of 0 to 7. This means the RL agent will be exposed to disruption scenarios 0-7. Scenario 8 will be left out to test the whether the RL agent has learnt a generalisable policy that can be applied to an unseen scenario. 

## Reinforcement Learning Agent

### Gymnasium Wrapper:

The dervied supply chain is then wrapped using the Gymnasium wrapper (see documentation: https://gymnasium.farama.org/api/wrappers/)

This acts as a standard enviroment interface, defining the discrete action space, how the simulation resets and reinitialises, how it steps through the 1-365 time steps in an episode (one supply chain simulation run) and how to store observations. 

The main benefit of using the Gymnasium environment wrapper is that you don't have to alter the underlying functions defining the supply chain, you simply use them in the wrapper. The resulting Gymnasium environment is then a compatiple input to the Stable Baselines3 reinforcement learning algorithm library which is what actually trains the agent.

```
pip install gymnasium
```

### Action Space

The RL agent controls:

**Supplier Allocation:**
```math
\alpha_t \in [0.2,0.5,0.8]
```

**Route Selection:**
$$route = \begin{cases} 0 & \text{for Route A} \\ 1 & \text{for Route B} \end{cases}$$

**Safety Stock:**

Safety stocks are a target net-stock that we will sometimes be above or below, but aim to on average achieve so that we may satisfy a desiered service level.

If inventory levels are normally distributed, the optimal safety stock can be calculated by the following formula:

```math
S^* = \phi^{-1}[fill rate]\sqrt{L\sigma^2_D}
```

Hence the action space should be something like this...

```math
S_t \in [20,29,115]
```
**Gymnasium Action Space**

The RL agent will choose a combination of the actions previously laid out in every step. Therefore, the actual action space is a cartesian product of all possible combination of actions, wich results in $3*2*3=18$ possible action combinations.

### Reward Function
In this supply chain we want to reward a high fullfilment rate but penalise a large NSAmp ratios, large backlogs and holding excessive inventory...
```math
R_t = 100*Fillrate_t -2NSAmp_t -0.5*B_t -0.1*I_t
```

### State Space

The observations vector generated by the supply chain simulation will contain:

* Step, t
* Inventory, $I_t$
* Backlog, $B_t$
* Net Stock, $Available_t$
* Demand, $D_t$
* Orders, $O_t$
* Supplier A Capacity, $C^A_t$
* Supplier B Capacity, $C^B_t$
* Route A Traffic, $Traffic^A_t$
* Route B Traffic, $Traffic^B_t$
* Rolling fill rate, $FillRate_t$
* Rolling NSAmp Ratio, $NSAmp_t$
* Demand noise, $demand_std$

Then the RL agent chooses an action vector at each step and the resulting reward will be calculated using the reward function. This will then form the data space to train the RL agent. 

### Stable Baseline3

Stable Baseline3 is a reinforcement learning library of algorithms. It requires python 3.1+ and PyTorch >= 2.8, documentation available at: https://stable-baselines3.readthedocs.io/en/master/index.html 

```
pip install "stable-baselines3>=2.0" rich 2>&1 | tail -5
```

### Deep Q-Network (DQN)

The Deep Q-Network (DQN) algorithm from Stable Baseline3 is used to train the RL agent. 

```
from stable_baselines3 import DQN
``` 
You then configure DQN by setting the following constants:

* TOTAL_TIMESTEPS which is the number of steps you do in total e.g. 292,000 represents 800 episodes of time steps from 1 to 365
* EVAL_FREQ is after how many time steps does it evaluate its learning e.g. 10,000 would mean it evaluates learning after 27 episodes
* N_EVAL_EPISODES is the number of episodes you use to evaluate whether the RL agent has learnt anything by getting it to follow its learnt optimal policy.
* SEED set the seed of the training for reproducibility

You then set-up two separate the environments, one for training and one for evaluation. In the context of this research the environment represents all the supply chain simulation observations stored as a normalised vector. The observations are normalised because their scales vary massively e.g. Fill Rate takes [0,1] where as NSAmp takes [1, 1000+]. Without normalisation the learning of weights would be dominated by the largest-scale observations. The evaluation environment serves as a separate environment to check whether the agent is learning e.g. after every 10,000 time steps 10 episodes are ran and thr agent choose the optimal learnt actions and the average reward is tracked.


Then DQN model itself takes the following parameters:

* env - is the supply chain simulation wrapped in the Gymnasium wrapper (SupplyChainEnv)
* Policy model to use, in this case the MLPolicy is used which is a multi-layer perceptron as the Q-network
* Learning rate this controls how much you change neural network weights in the direction of the calculate gradient of the loss function
* Learning starts is the step number you stop choosing random actions and start learning optimal actions
* Buffer size holds the replay buffer of the last action, reward, next-state tranistions you then sample mini-batches from, e.g. it holds the last 50,000 time steps and then you sample from them to calculate loss and update weights
* Batch size is the number of samples in the mini batch sampled from the replay buffer. These are then used to calculate an average loss
* Gamma is the discount factor you apply to previous rewards
* Train frequency is the gradient update per environment steps
* Target update interval is the number of steps until you update the target network in comparison to the online network
* Exploration fraction is the starting epsilon probabilty for choosing a random action
* Exploration final eplsilon is the final probability you choose a random action
* Policy kwargs is the architecture of the Q-network i.e. the number of neurons and hidden layers

### Evaluation Call Back

The frequency of evaluation is set and then say every 10,000 training steps copies of the current statistics from the training environment are pasted into the evaluation environment. Then 10 episodes are run using the optimal policy learnt and then if the mean reward is a new best, it saves the modle weights to a zip file and stores the evaluation results in a file to be used to create the learning curve plot.

## Testing 

The trained DQN agent will be compared using a time-series of the reward function, Fill Rate and NSAmp Ratio against:

* Fixed policy - do nothing
* Random policy
* Rule-based heuristic policy
* DQN policy

Finally, to test whether the agent has learnt a generalisable policy it will be evaluated against a simulation where all three disruptions occur randomly.  

## Streamlit app

A separate, interactive app that optimises a single manufacturer under normal conditions (no disruptions). It doesn't use the DQN code.

| File | Contents |
|---|---|
| [`sc_model.py`](sc_model.py) | Simulation, optimisation and theory (numpy/scipy only). A comment block at the top lists every difference from `supply_chain_sim.py`. |
| [`supply_chain_simulation_app.py`](supply_chain_simulation_app.py) | The Streamlit UI. |
| [`tests/test_sc_model.py`](tests/test_sc_model.py) | Sanity tests against theory and a brute-force optimiser. |

### Running it

```bash
pip install -r requirements.txt
streamlit run supply_chain_simulation_app.py
```

`requirements.txt` lists only the app's dependencies (streamlit, numpy, pandas, scipy, plotly), which keeps a deployment well under 1 GB. The DQN project still installs its own dependencies with the `pip install` command under [Installation](#installation). To run the tests, also install `pytest` and run `pytest tests/`.

### What it demonstrates

The manufacturer orders with the POUT policy from above, generalised to a review cadence of c days (daily, weekly or monthly) and random lead times L ~ max(1, round(N(μ_L, σ_L))):

```math
O_t = max(0, round(c\mu_D + \frac{1}{T}(ss - NS_t) + \frac{1}{T}(\mu_L\mu_D - WIP_t)))
```

Here $NS_t$ is net stock (inventory − backlog) and $WIP_t$ is the sum of orders placed but not yet received. Each order is produced evenly over the next c days. Daily cost is holding $h \cdot I$ + backlog $b \cdot B$ + capacity $u \cdot k$ + overtime $w \cdot max(0, P-k)$. Every unit of capacity costs u per day, used or not, and every unit produced above capacity costs w.

1. **The T trade-off.** Small T reacts hard to every deviation. It holds net stock tight (low NSAmp) but amplifies orders (Bullwhip = 1/(2T−1)). Large T smooths production but lets inventory drift. The sweep chart shows both against T, along with the resulting cost.
2. **Safety stock (ss)** buffers demand and lead-time uncertainty. The optimum is the smallest ss that meets the target fill rate, or the newsvendor quantile b/(b+h) of net stock if that is larger. It is shown next to King's formula.
3. **Capacity (k)** buffers order variability. Adding a unit of capacity costs u every day and saves w on overtime days, so the cost-minimising k is the (1 − u/w) quantile of daily production, which matches μ_D + z·σ_P. It sits below mean production when overtime costs less than twice the capacity cost (w < 2u), and above it when w > 2u.
4. **Cadence damping.** Ordering every c days spreads each order over c days of production, so daily production variance falls by a factor of c. The trade-off is more cycle stock to hold.

**Optimise** finds (ss\*, k\*, T\*). For each T on a 0.1–10 grid it simulates 30 replications of 365 days (after a warm-up) with common random numbers, then refines T* with a bounded scalar search. Orders don't depend on ss, so ss and k are solved in closed form or by bisection for each T. **Edit** unlocks the parameters, and **Run** simulates one year with them, shows the cost against the optimum on the same random numbers, and offers the run as a CSV.
