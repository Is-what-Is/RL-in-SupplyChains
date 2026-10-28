"""
Streamlit UI for the supply chain optimisation app.
All simulation, optimisation and theory lives in sc_model.py.

Run with:  streamlit run supply_chain_simulation_app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sc_model as m


st.set_page_config(page_title="Supply chain optimisation", layout="wide")

# Categorical slots in fixed order (validated reference palette)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY = "#8a8984"
UNSTABLE_FILL = "rgba(138, 137, 132, 0.18)"

DEFAULTS = {
    "mu_d": 50, "sigma_d": 10.0, "mu_L": 7, "sigma_L": 2.0, "fill_target": 0.95,
    "h": 1.0, "b": 9.0, "u": 2.0, "w": 4.0, "seed": 42, "cadence": "Daily",
    "ss": 100, "k": 50.0, "T": 2.0, "edit_mode": False, "opt": None, "run": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_optimise(mu_d, sigma_d, mu_L, sigma_L, fill_target, h, b, u, w, c, seed):
    inputs = m.SCInputs(mu_d, sigma_d, mu_L, sigma_L, fill_target, h, b, u, w)
    return m.optimise(inputs, c, seed=seed)


def current_inputs():
    s = st.session_state
    return m.SCInputs(float(s.mu_d), float(s.sigma_d), float(s.mu_L), float(s.sigma_L),
                      float(s.fill_target), float(s.h), float(s.b), float(s.u), float(s.w))


def clamp_state(key, lo, hi):
    st.session_state[key] = float(min(max(st.session_state[key], lo), hi))


def toggle_edit():
    st.session_state.edit_mode = not st.session_state.edit_mode


def ranged_input(label, caption, key, **kwargs):
    st.caption(caption)
    return st.number_input(label, key=key, **kwargs)


def sweep_figure(opt):
    sw = opt.sweep
    stable = sw[sw["T"] >= 0.6]
    custom = np.column_stack([sw["ss_star_T"], sw["k_star_T"], sw["fill_rate"]])
    hover = ("T = %{x:.1f}<br>%{fullData.name} = %{y:,.3f}"
             "<br>ss*(T) = %{customdata[0]:.0f}<br>k*(T) = %{customdata[1]:.2f}"
             "<br>fill rate = %{customdata[2]:.3f}<extra></extra>")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Variance ratios (log scale)",
                                        "Expected total cost at (ss*(T), k*(T))"))
    fig.add_trace(go.Scatter(x=sw["T"], y=sw["bullwhip"], name="Bullwhip (simulated)",
                             line=dict(color=BLUE, width=2), customdata=custom,
                             hovertemplate=hover, legend="legend"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sw["T"], y=sw["bullwhip_theory"], name="Bullwhip theory 1/(2T−1)",
                             line=dict(color=BLUE, width=2, dash="dash"), customdata=custom,
                             hovertemplate=hover, legend="legend"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sw["T"], y=sw["nsamp"], name="NSAmp (simulated)",
                             line=dict(color=ORANGE, width=2), customdata=custom,
                             hovertemplate=hover, legend="legend"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sw["T"], y=sw["total_cost"], name="Total cost",
                             line=dict(color=AQUA, width=2), customdata=custom,
                             hovertemplate=hover, showlegend=False), row=2, col=1)

    ratios = np.concatenate([stable["bullwhip"], stable["nsamp"]])
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    fig.update_yaxes(type="log", row=1, col=1, title_text="Var ratio", tickvals=ticks,
                     range=[np.log10(ratios.min() * 0.7), np.log10(ratios.max() * 1.5)])
    costs = sw.loc[sw["T"] >= 0.8, "total_cost"]
    pad = 0.05 * (costs.max() - costs.min()) + 1
    fig.update_yaxes(row=2, col=1, title_text="Cost over horizon",
                     range=[costs.min() - pad, costs.max() + pad])
    fig.update_xaxes(range=[0, 10.05], row=2, col=1, title_text="Proportional feedback T")

    fig.add_vrect(x0=0, x1=0.5, fillcolor=UNSTABLE_FILL, line_width=0, row="all", col=1)
    fig.add_annotation(x=0.25, y=0, yref="y2 domain", text="unstable<br>T ≤ 0.5", showarrow=False,
                       font=dict(size=11, color=GREY), yanchor="bottom")
    fig.add_vline(x=opt.T_star, line=dict(color=GREY, width=1.5, dash="dot"), row="all", col=1)
    fig.add_annotation(x=opt.T_star, y=1, yref="y2 domain", text=f"T* = {opt.T_star:.2f}",
                       showarrow=False, xanchor="left", yanchor="top", xshift=4,
                       font=dict(size=11, color=GREY))
    fig.update_layout(height=620, hovermode="x unified", margin=dict(t=40, b=40, l=10, r=10),
                      legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"))
    return fig


def run_figure(df, c, k, fill_target):
    titles = ("Demand, orders and production (units/day)", "Inventory and backlog",
              "Cumulative fill rate", "Daily cost by component")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=titles, row_heights=[1.2, 1, 0.7, 1])

    def add(row, legend, y, name, color, **kw):
        fig.add_trace(go.Scatter(x=df["day"], y=y, name=name, legend=legend,
                                 line=dict(color=color, width=kw.pop("width", 1.5)), **kw),
                      row=row, col=1)

    review = df["order"].where(df["lead_time_of_order"].notna())
    add(1, "legend", df["demand"], "Demand", BLUE)
    add(1, "legend", df["production"], "Production", ORANGE, width=2)
    fig.add_trace(go.Scatter(x=df["day"], y=review / c, name="Order ÷ c (review days)",
                             mode="markers", marker=dict(color=AQUA, size=8, line=dict(width=2, color="white")),
                             customdata=np.column_stack([df["order"], df["lead_time_of_order"]]),
                             hovertemplate="order %{customdata[0]:.0f}, lead time %{customdata[1]:.0f}d<extra></extra>",
                             legend="legend"), row=1, col=1)
    fig.add_hline(y=k, line=dict(color=GREY, dash="dash", width=1.5), row=1, col=1,
                  annotation_text=f"capacity k = {k:.1f}", annotation_position="top left",
                  annotation_font_color=GREY)

    add(2, "legend2", df["inventory"], "Inventory", BLUE)
    add(2, "legend2", df["backlog"], "Backlog", ORANGE)

    add(3, "legend3", df["fill_rate_cumulative"], "Fill rate", BLUE, width=2, showlegend=False)
    fig.add_hline(y=fill_target, line=dict(color=GREY, dash="dash", width=1.5), row=3, col=1,
                  annotation_text=f"target {fill_target:.2f}", annotation_position="bottom left",
                  annotation_font_color=GREY)

    for name, color in [("holding_cost", BLUE), ("backlog_cost", ORANGE),
                        ("reserved_capacity_cost", AQUA), ("overtime_cost", YELLOW)]:
        add(4, "legend4", df[name], name.replace("_cost", "").replace("_", " ").capitalize(), color,
            stackgroup="cost", width=0.5)

    # One legend per subplot, pinned to the top-right of its row
    for row, legend in [(1, "legend"), (2, "legend2"), (4, "legend4")]:
        axis = fig.layout["yaxis" if row == 1 else f"yaxis{row}"]
        fig.update_layout({legend: dict(x=1, xanchor="right", y=axis.domain[1], yanchor="top",
                                        orientation="h", traceorder="normal", bgcolor="rgba(0,0,0,0)")})
    fig.update_xaxes(title_text="Day (after warm-up)", row=4, col=1)
    fig.update_layout(height=900,hovermode="x unified", margin=dict(t=40, b=40, l=10, r=10))
    return fig


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------

st.title("Supply chain optimisation: POUT ordering under normal conditions")
st.caption("Trade inventory variance against order variance with the feedback parameter T, "
           "buffer uncertainty with safety stock and order variability with capacity, and "
           "see how the ordering cadence damps production.")

left, right = st.columns([1, 2], gap="large")

with left:
    # 1. Input variables -------------------------------------------------------
    with st.container(border=True):
        st.subheader("Input variables")

        mu_d = ranged_input("Mean demand (units/day)", "Range: 10–100", "mu_d",
                            min_value=10, max_value=100, step=1)
        sd_lo, sd_hi = round(0.1 * mu_d, 2), round(0.5 * mu_d, 2)
        clamp_state("sigma_d", sd_lo, sd_hi)
        ranged_input("Demand std", f"Range: {sd_lo:g}–{sd_hi:g} (0.1–0.5 × mean)", "sigma_d",
                     min_value=sd_lo, max_value=sd_hi, step=0.5)

        mu_L = ranged_input("Mean lead time (days)", "Range: 5–25 days", "mu_L",
                            min_value=5, max_value=25, step=1)
        sl_lo, sl_hi = round(0.2 * mu_L, 2), round(0.4 * mu_L, 2)
        clamp_state("sigma_L", sl_lo, sl_hi)
        ranged_input("Lead time std (days)", f"Range: {sl_lo:g}–{sl_hi:g} (0.2–0.4 × mean lead time)",
                     "sigma_L", min_value=sl_lo, max_value=sl_hi, step=0.1)

        ranged_input("Target fill rate", "Range: 0.75–0.99", "fill_target",
                     min_value=0.75, max_value=0.99, step=0.01, format="%.2f")

        col_a, col_b = st.columns(2)
        with col_a:
            ranged_input("Holding cost h", "Range: 1–10 per unit/day", "h", min_value=1.0, max_value=10.0, step=0.5)
            ranged_input("Capacity cost u", "Range: 1–10 per unit of capacity/day, u < w", "u", min_value=1.0, max_value=10.0, step=0.5)
        with col_b:
            ranged_input("Backlog cost b", "Range: 1–10 per unit/day", "b", min_value=1.0, max_value=10.0, step=0.5)
            ranged_input("Overtime cost w", "Range: 1–10 per overtime unit, w > u", "w", min_value=1.0, max_value=10.0, step=0.5)

        costs_valid = st.session_state.u < st.session_state.w
        if not costs_valid:
            st.error("Capacity cost u must be less than the overtime cost w.")

        with st.expander("Advanced"):
            st.number_input("Random seed", key="seed", min_value=0, step=1)

        optimise_clicked = st.button("Optimise", type="primary", disabled=not costs_valid,
                                     width="stretch", key="optimise")

    # 2. Cadence ---------------------------------------------------------------
    cadence = st.selectbox("Cadence of the supply chain", list(m.CADENCES), key="cadence")
    c = m.CADENCES[cadence]
    st.caption(f"Orders are placed every {c} day{'s' if c > 1 else ''}.")

    inputs = current_inputs()
    opt_key = (inputs, c, int(st.session_state.seed))

    if optimise_clicked:
        with st.spinner("Optimising ss, k and T…"):
            opt = cached_optimise(*[getattr(inputs, f) for f in
                                    ("mu_d", "sigma_d", "mu_L", "sigma_L", "fill_target", "h", "b", "u", "w")],
                                  c, int(st.session_state.seed))
        st.session_state.opt = {"result": opt, "key": opt_key, "cadence": cadence}
        st.session_state.ss = int(opt.ss_star)
        st.session_state.k = round(float(opt.k_star), 2)
        st.session_state.T = round(float(opt.T_star), 2)
        st.session_state.run = None

    opt_state = st.session_state.opt
    opt_current = opt_state is not None and opt_state["key"] == opt_key

    # 3. Parameters ------------------------------------------------------------
    with st.container(border=True):
        head, edit_col = st.columns([5, 1], vertical_alignment="center")
        head.subheader("Parameters")
        edit_col.button("Lock" if st.session_state.edit_mode else "Edit", key="edit",
                        on_click=toggle_edit)
        locked = not st.session_state.edit_mode

        th = m.theory(inputs, c, float(st.session_state.T))
        st.caption(
            f"Theoretical safety stock (King, OUT T=1) = z × √((L+R)σd² + (σL·μd)²) = "
            f"{th['z_ss']:.3f} × √(({inputs.mu_L:g}+{th['R']})·{inputs.sigma_d:g}² + "
            f"({inputs.sigma_L:g}·{inputs.mu_d:g})²) = **{th['ss_theory']:.1f}**")
        st.number_input("Safety stock ss (units)", key="ss", step=1, disabled=locked)

        if th["k_theory"] is None:
            k_caption = (f"Theoretical capacity = μd + z·σP: undefined, unstable at T = "
                         f"{st.session_state.T:.2f} (needs T > 0.5)")
        else:
            k_caption = (f"Theoretical capacity = μd + z·σP = **{th['k_theory']:.2f}** "
                         f"(z = Φ⁻¹(1 − u/w) = {th['z_k']:.3f}, σP = {th['sigma_P']:.2f} at T = "
                         f"{st.session_state.T:.2f})")
        st.caption(k_caption)
        st.number_input("Capacity k (units/day)", key="k", min_value=0.0, step=0.5, disabled=locked)

        st.caption("Range: 0.1–10 (T ≤ 0.5 is unstable)")
        st.number_input("Proportional feedback T", key="T", min_value=0.1, max_value=10.0,
                        step=0.1, format="%.2f", disabled=locked)

        if opt_state is not None:
            opt = opt_state["result"]
            if not opt_current:
                st.warning("Inputs, cadence or seed changed since the last optimisation. "
                           "Press Optimise to refresh.")
            m1, m2 = st.columns(2)
            m1.metric("Optimal expected cost", f"{opt.cost:,.0f}")
            m2.metric("Achieved fill rate", f"{opt.fill_rate:.3f}")
            th_opt = m.theory(opt_state["key"][0], opt.c, opt.T_star)
            st.table({
                "": ["Safety stock ss", "Capacity k"],
                "Simulated optimum": [f"{opt.ss_star}", f"{opt.k_star:.2f}"],
                "Theory": [f"{th_opt['ss_theory']:.1f}", f"{th_opt['k_theory']:.2f}"],
            })
            st.caption("They differ because King's formula targets a cycle service level, not "
                       "the fill rate (which needs less stock), assumes T = 1 rather than "
                       f"T* = {opt.T_star:.2f}, and ignores integer rounding and clipping of "
                       "demand, lead times and orders.")

        run_clicked = st.button("Run", key="run_btn", width="stretch")

    if run_clicked:
        ss, k, T = int(st.session_state.ss), float(st.session_state.k), float(st.session_state.T)
        seed = int(st.session_state.seed)
        df = m.simulate_run(inputs, c, ss, k, T, seed=seed)
        run = {"df": df, "summary": m.run_summary(df, c), "ss": ss, "k": k, "T": T,
               "cadence": cadence, "fill_target": inputs.fill_target, "opt_diff": None}
        if opt_current:
            opt = opt_state["result"]
            if (ss, round(k, 2), round(T, 2)) != (opt.ss_star, round(opt.k_star, 2), round(opt.T_star, 2)):
                opt_df = m.simulate_run(inputs, c, opt.ss_star, opt.k_star, opt.T_star, seed=seed)
                run["opt_diff"] = run["summary"]["total_cost"] - float(opt_df["total_cost"].sum())
        st.session_state.run = run


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------

with right:
    if st.session_state.opt is None and st.session_state.run is None:
        st.info("Press **Optimise** to sweep T and find the cost-minimising (ss, k, T), "
                "or **Edit** the parameters and press **Run** to simulate one year.")

    if st.session_state.opt is not None:
        opt = st.session_state.opt["result"]
        st.subheader(f"Optimisation sweep over T ({st.session_state.opt['cadence'].lower()} cadence)")
        st.caption(f"ss* = {opt.ss_star}, k* = {opt.k_star:.2f}, T* = {opt.T_star:.2f}; "
                   f"Bullwhip = {opt.bullwhip:.3f}, NSAmp = {opt.nsamp:.2f}. "
                   f"Each point is {m.N_REPS} replications of {m.SCInputs().horizon} days with "
                   "common random numbers (seed "
                   f"{opt.seed}). Lower T damps inventory (NSAmp) but amplifies orders (Bullwhip).")
        st.plotly_chart(sweep_figure(opt), width="stretch")

    run = st.session_state.run
    if run is not None:
        s = run["summary"]
        st.subheader(f"Simulation run ({run['cadence'].lower()} cadence, "
                     f"ss = {run['ss']}, k = {run['k']:.2f}, T = {run['T']:.2f})")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total cost", f"{s['total_cost']:,.0f}",
                  None if run["opt_diff"] is None else f"{run['opt_diff']:+,.0f} vs optimum",
                  delta_color="inverse")
        r2.metric("Fill rate", f"{s['fill_rate']:.3f}")
        r3.metric("Bullwhip", f"{s['bullwhip']:.3f}")
        r4.metric("NSAmp", f"{s['nsamp']:.2f}")
        if run["opt_diff"] is not None:
            st.caption("Cost difference against the optimal (ss*, k*, T*) run on the same random numbers.")
        st.plotly_chart(run_figure(run["df"], m.CADENCES[run["cadence"]], run["k"], run["fill_target"]), width="stretch")
        st.download_button(
            "Download run as CSV", run["df"].to_csv(index=False).encode(),
            file_name=f"sc_run_{run['cadence'].lower()}_ss{run['ss']}_k{run['k']:.2f}_T{run['T']:.2f}.csv",
            mime="text/csv", key="download")
