# simulation.py

import math
import numpy as np
from scipy.integrate import solve_ivp

from model import Model
import state  # ApexState, SLIPState, TouchDownState, Phase

import matplotlib.pyplot as plt
from matplotlib import animation


# ------------------------------------------------------------
# 1. Animation utility
# ------------------------------------------------------------
def animate_slip_step(
    t_array,
    x_array,
    y_array,
    foot_x_array,
    is_stance,
    l0,
    theta,
):
    """
    Animate a multi-step SLIP trajectory.

    t_array      : 1D array (global time)
    x_array      : 1D array (COM x)
    y_array      : 1D array (COM y)
    foot_x_array : 1D array (foot x at each frame; NaN or anything during flight)
    is_stance    : 1D bool array (True where stance is active)
    l0           : float (rest leg length)
    theta        : float (leg angle at touchdown, from vertical)
    """

    t_array      = np.asarray(t_array)
    x_array      = np.asarray(x_array)
    y_array      = np.asarray(y_array)
    foot_x_array = np.asarray(foot_x_array)
    is_stance    = np.asarray(is_stance, dtype=bool)

    fig, ax = plt.subplots()

    margin_x        = 3.0
    margin_y_top    = 3.0
    margin_y_bottom = 0.1

    x_min = min(x_array.min(), np.nanmin(foot_x_array)) - margin_x
    x_max = max(x_array.max(), np.nanmax(foot_x_array)) + margin_x
    y_min = -margin_y_bottom
    y_max = y_array.max() + margin_y_top

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", "box")

    # Ground
    ax.axhline(0.0, color="black", linewidth=2)

    # Ball + bar
    ball, = ax.plot([], [], "o", color="red", markersize=8)
    bar,  = ax.plot([], [], "-", color="blue", linewidth=2)

    # State text
    state_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_title("SLIP Multi-Step Animation")

    def init():
        ball.set_data([], [])
        bar.set_data([], [])
        state_text.set_text("")
        return ball, bar, state_text

    def update(frame_idx):
        x = x_array[frame_idx]
        y = y_array[frame_idx]

        # Ball at COM
        ball.set_data([x], [y])

        if is_stance[frame_idx]:
            label = "STANCE"
            fx = foot_x_array[frame_idx]
            bar.set_data([fx, x], [0.0, y])
        else:
            label = "FLIGHT"
            # flight leg: attached to COM with fixed geometry
            x_foot_flight = x + l0 * math.sin(theta)
            y_foot_flight = y - l0 * math.cos(theta)
            bar.set_data([x_foot_flight, x], [y_foot_flight, y])

        state_text.set_text(label)
        return ball, bar, state_text

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(t_array),
        init_func=init,
        interval=60,
        blit=True,
    )

    plt.show()
    return ani


# ------------------------------------------------------------
# 2. Initial apex helper
# ------------------------------------------------------------
def start_apex(x: float = 0.0, y: float = 2.0, xdot: float = 0.0, step_idx: int = 0):
    """
    Build the initial apex state and return (ApexState, [x, y, xdot, ydot]).
    """
    apex0    = state.ApexState(x=x, y=y, xdot=xdot, step_idx=step_idx)
    s0_state = apex0.to_full_state()   # ydot = 0, phase = FLIGHT
    s0_vec   = s0_state.as_vector()    # [x, y, xdot, ydot]
    return apex0, s0_vec


# ------------------------------------------------------------
# 3. One-step simulation: apex -> TD -> stance -> LO -> next apex
# ------------------------------------------------------------
def simulate_one_step(
    model: Model,
    apex: state.ApexState,
    method: str = "RK45",
    t_max: float = 5.0,
):
    """
    Simulate a single SLIP step:

        apex -> flight1 (touchdown) -> stance (liftoff) -> flight2 (next apex)

    Returns:
        apex_next        : ApexState at next apex (or None if step fails)
        t_array          : stacked time (flight1 + stance + flight2) [local time]
        x_array, y_array : COM positions
        is_stance        : bool array (True where stance is active)
        foot_x_array     : stance foot x, same length as t_array
    """

    # ---------- Initial apex -> full state ----------
    s0_state = apex.to_full_state()
    s0_vec   = s0_state.as_vector()

    # ---------- 1) FLIGHT 1: apex -> touchdown ----------
    def event_touchdown(t, s):
        x, y, xdot, ydot = s
        # touchdown when COM reaches y = l0 cos(theta) while descending
        return y - model.l0 * math.cos(model.theta)

    event_touchdown.terminal  = True
    event_touchdown.direction = -1

    sol_f1 = solve_ivp(
        fun=model.flight_dynamics,
        t_span=(0.0, t_max),
        y0=s0_vec,
        method=method,
        events=event_touchdown,
        max_step=0.01,
    )

    if sol_f1.t_events[0].size == 0:
        print(f"[simulate_one_step] No touchdown found at step {apex.step_idx}.")
        return None

    t_f1    = sol_f1.t
    x_f1    = sol_f1.y[0]
    y_f1    = sol_f1.y[1]
    xdot_f1 = sol_f1.y[2]
    ydot_f1 = sol_f1.y[3]

    # touchdown time and state from event
    td_time = sol_f1.t_events[0][0]
    td_vec  = sol_f1.y_events[0][0]   # [x, y, xdot, ydot] at touchdown

    flight_td_state = state.SLIPState.from_vector(
        td_vec,
        phase=state.Phase.FLIGHT,
        t=td_time,
        step_idx=apex.step_idx,
    )

    touchdown = state.TouchDownState.from_flight_state(
        slip_state=flight_td_state,
        theta=model.theta,
        l0=model.l0,
    )

    foot_x    = touchdown.foot_x
    s0_stance = touchdown.as_vector()     # [x_td, y_td, xdot_td, ydot_td]

    # ---------- 2) STANCE: touchdown -> liftoff ----------
    def stance_dyn(t, s):
        return model.stance_dynamics(t, s, foot_x)

    def event_liftoff(t, s):
        x, y, xdot, ydot = s
        dx = x - foot_x
        l  = math.hypot(dx, y)
        # liftoff when leg returns to rest length
        return l - model.l0

    event_liftoff.terminal  = True
    event_liftoff.direction = +1

    sol_st = solve_ivp(
        fun=stance_dyn,
        t_span=(td_time, td_time + t_max),
        y0=s0_stance,
        method=method,
        events=event_liftoff,
        max_step=0.01,
    )

    if sol_st.t_events[0].size == 0:
        print(f"[simulate_one_step] No liftoff found at step {apex.step_idx}.")
        return None

    t_st    = sol_st.t
    x_st    = sol_st.y[0]
    y_st    = sol_st.y[1]
    xdot_st = sol_st.y[2]
    ydot_st = sol_st.y[3]

    lo_time = sol_st.t_events[0][0]
    lo_vec  = sol_st.y_events[0][0]   # [x, y, xdot, ydot] at liftoff

    # ---------- 3) FLIGHT 2: liftoff -> next apex ----------
    def event_apex(t, s):
        x, y, xdot, ydot = s
        return ydot  # vertical velocity zero at apex

    event_apex.terminal  = True
    event_apex.direction = -1  # crossing from up to down

    sol_f2 = solve_ivp(
        fun=model.flight_dynamics,
        t_span=(lo_time, lo_time + t_max),
        y0=lo_vec,
        method=method,
        events=event_apex,
        max_step=0.01,
    )

    t_f2    = sol_f2.t
    x_f2    = sol_f2.y[0]
    y_f2    = sol_f2.y[1]
    xdot_f2 = sol_f2.y[2]
    ydot_f2 = sol_f2.y[3]

    apex_time = sol_f2.t_events[0][0]
    apex_vec  = sol_f2.y_events[0][0]   # [x, y, xdot, 0] at next apex

    apex_next = state.ApexState(
        x=apex_vec[0],
        y=apex_vec[1],
        xdot=apex_vec[2],
        step_idx=apex.step_idx + 1,
    )

    # ---------- Stack trajectories (flight1 + stance + flight2) ----------
    t_array = np.concatenate([t_f1,      t_st,     t_f2])
    x_array = np.concatenate([x_f1,      x_st,     x_f2])
    y_array = np.concatenate([y_f1,      y_st,     y_f2])

    len_f1 = len(t_f1)
    len_st = len(t_st)
    len_f2 = len(t_f2)

    is_stance = np.zeros(len_f1 + len_st + len_f2, dtype=bool)
    is_stance[len_f1 : len_f1 + len_st] = True

    # foot x per frame
    foot_x_array = np.full_like(t_array, np.nan, dtype=float)
    foot_x_array[len_f1 : len_f1 + len_st] = foot_x

    return apex_next, t_array, x_array, y_array, is_stance, foot_x_array


# ------------------------------------------------------------
# 4. Multi-step simulation
# ------------------------------------------------------------
def simulate_multiple_steps(
    model: Model,
    apex0: state.ApexState,
    n_steps: int,
    method: str = "RK45",
    t_max_per_step: float = 5.0,
):
    """
    Run several SLIP steps in sequence.

    Returns:
        all_t, all_x, all_y, all_stance, all_foot_x
    """
    apex = apex0

    all_t      = None
    all_x      = None
    all_y      = None
    all_stance = None
    all_foot_x = None

    t_offset = 0.0

    for k in range(n_steps):
        res = simulate_one_step(model, apex, method=method, t_max=t_max_per_step)
        if res is None:
            print(f"[simulate_multiple_steps] Stopping at step {k} (step failed).")
            break

        apex_next, t_step, x_step, y_step, stance_step, foot_x_step = res

        # Shift local step time to global time by adding offset
        t_step_global = t_step + t_offset

        if all_t is None:
            # first step
            all_t      = t_step_global
            all_x      = x_step
            all_y      = y_step
            all_stance = stance_step
            all_foot_x = foot_x_step
        else:
            all_t      = np.concatenate([all_t,      t_step_global[1:]])
            all_x      = np.concatenate([all_x,      x_step[1:]])
            all_y      = np.concatenate([all_y,      y_step[1:]])
            all_stance = np.concatenate([all_stance, stance_step[1:]])
            all_foot_x = np.concatenate([all_foot_x, foot_x_step[1:]])

        t_offset = all_t[-1]
        apex     = apex_next
        print(f"Step {k} end apex:", apex_next)

    return all_t, all_x, all_y, all_stance, all_foot_x


# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
def main():
    model = Model()
    method = "RK45"

    # How many steps to simulate
    n_steps = 5
    t_max_per_step = 5.0  # "big enough" window for each step's events

    # Initial apex
    apex0, _ = start_apex(x=0.0, y=2.0, xdot=0.0, step_idx=0)

    # Multi-step simulation
    t_array, x_array, y_array, is_stance, foot_x_array = simulate_multiple_steps(
        model=model,
        apex0=apex0,
        n_steps=n_steps,
        method=method,
        t_max_per_step=t_max_per_step,
    )

    # Animate full trajectory
    animate_slip_step(
        t_array,
        x_array,
        y_array,
        foot_x_array,
        is_stance,
        model.l0,
        model.theta,
    )

if __name__ == "__main__":
    main()
