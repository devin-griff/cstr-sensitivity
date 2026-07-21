# =============================================================================
# CSTR Sensitivity: a Streamlit tutorial app.
#
# Local feedback gains from one solve. The app solves an optimal control
# problem for the Hicks-Ray CSTR from a user-chosen initial state, then reads
# the local gain matrix out of the KKT factorization the solver is already
# holding: each gain is one backsolve, no extra solver run. From the same
# factorization, estimate() predicts the entire re-optimized solution at a
# perturbed initial state in microseconds; an exact re-solve (on a clone, so
# the held factorization stays with the baseline) shows how close the
# first-order prediction landed and what the exact answer costs instead.
#
# Library roadmap:
#   - streamlit    : UI framework. Each interaction reruns this script
#                     top-to-bottom; persistent values live in session_state.
#   - pyomo        : algebraic modeling, with pyomo.dae for the collocation
#                     discretization of the reactor ODEs.
#   - pyomo-cvp    : piecewise-constant control parameterization: declare
#                     profiles before discretization, parameterize after.
#   - pyomo-pounce : the NLP solver (Rust reimplementation of IPOPT) plus
#                     the sIPOPT-style sensitivity system: declare_sens_param,
#                     gradient, estimate.
#   - matplotlib   : static figures (Wong color palette).
#
# File roadmap:
#   1. Page config + CSS.
#   2. Sidebar: initial-condition sliders + Solve, perturbed-start sliders
#      + Estimate & Re-solve (shown once a baseline solve exists).
#   3. build_model / solve_baseline / run_perturbation: the computation.
#   4. Figure builders: baseline and perturbed time series and phase
#      planes on fixed axes ([0, 1] everywhere except the phase plots'
#      temperature, which spans the realistic trajectory envelope); the
#      static schematic ships pre-rendered as schematic.png.
#   5. render_formulation_tab: static markdown reference material.
#   6. Main layout: four tabs (Time Series, Phase Plot, Formulation, Logs).
# =============================================================================

import atexit
import base64
import contextlib
import gc
import io
import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo_cvp import declare_profile
# pounce: importing the package registers the bundled solver binary with
# Pyomo's SolverFactory, so pyo.SolverFactory("pounce") resolves below.
import pyomo_pounce  # noqa: F401
from pyomo_pounce import declare_sens_param, estimate, gradient

# `set_page_config` must be the first Streamlit call.
st.set_page_config(page_title="CSTR Sensitivity", page_icon="favicon.png",
                   layout="wide", initial_sidebar_state="expanded")

# Wong colorblind-safe palette, shared across the app family.
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

# Model dimensions and steady-state targets, shared between the Pyomo model
# and the figures. The target is the open-loop unstable steady state of the
# dimensionless Hicks-Ray model (Huang, Patwardhan & Biegler, 2012).
N, H = 100, 1
ZC_SS, ZT_SS = 0.6416, 0.5387
V1_SS, V2_SS = 0.57828, 0.49989

# ── CSS ──────────────────────────────────────────────────────────────────────
# Sidebar-app variant of the family CSS: in-flow home logo, hidden sidebar
# header chrome, tightened paddings, and no cursor change on disabled
# buttons (disabled-ness here means "nothing new to compute", not an error).
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    user-select: none;
    -webkit-user-select: none;
}
section[data-testid="stSidebar"] > div:last-child,
[data-testid="stSidebarUserContent"] {
    padding-bottom: 0.5rem !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled,
section[data-testid="stSidebar"] .stButton > button:disabled:hover {
    cursor: default !important;
}
.home-logo-corner {
    display: inline-block;
    margin: 0 0 0.75rem;
}
.home-logo-corner img {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: block;
}
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-bottom: -0.25rem !important;
}
/* The gains chart: st.pyplot stretches images to the sidebar width, so
   dragging the sidebar wider would grow the chart past its fixed
   190px box and clip it. Cap the height instead (width follows the
   aspect ratio); a narrow sidebar binds on max-width and scales down. */
section[data-testid="stSidebar"] [data-testid="stImage"] img {
    max-height: 182px;
    max-width: 100%;
    width: auto !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
    padding-bottom: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
# Home link: the Griffith PSE logo navigates back to the portfolio site.
# Embedded from the local favicon.png as a base64 data URL so loading the
# page makes no third-party request.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.sidebar.markdown(
    f'<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE: home" />'
    f'</a>',
    unsafe_allow_html=True,
)

# Both control sections render further down, after their click handlers'
# dependencies exist: each is a fragment (moving its sliders reruns only
# that section, not the whole page), and the clicks are handled inside.

# ── Model + computation ──────────────────────────────────────────────────────

def build_model(zc0, zt0):
    """The Hicks-Ray CSTR optimal control problem, discretized and
    control-parameterized, with the initial condition declared as the
    sensitivity parameters. Identical to the companion notebook."""
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * H, H))

    m.u1sf = pyo.Param(initialize=600)    # coolant-flow scale factor
    m.u2sf = pyo.Param(initialize=40)     # residence-time scale factor
    m.k0 = pyo.Param(initialize=300)      # Arrhenius pre-exponential
    m.ea = pyo.Param(initialize=5)        # dimensionless activation energy
    m.a0 = pyo.Param(initialize=1.95e-4)  # heat-transfer coefficient
    m.ztcw = pyo.Param(initialize=0.38)   # coolant temperature
    m.ztf = pyo.Param(initialize=0.395)   # feed temperature

    m.zc_ss = pyo.Param(initialize=ZC_SS)  # steady-state targets
    m.zt_ss = pyo.Param(initialize=ZT_SS)
    m.v1_ss = pyo.Param(initialize=V1_SS)
    m.v2_ss = pyo.Param(initialize=V2_SS)

    # the initial condition: mutable, and the sensitivity parameters
    m.zc0 = pyo.Param(initialize=zc0, mutable=True)
    m.zt0 = pyo.Param(initialize=zt0, mutable=True)

    m.zc = pyo.Var(m.t, bounds=(0, 1), initialize=ZC_SS)
    m.zt = pyo.Var(m.t, bounds=(0, None), initialize=ZT_SS)
    m.dzc = DerivativeVar(m.zc, wrt=m.t)
    m.dzt = DerivativeVar(m.zt, wrt=m.t)
    m.v1 = pyo.Var(m.t, bounds=(0.166666666666667, 1), initialize=V1_SS)
    m.v2 = pyo.Var(m.t, bounds=(0.025, 1), initialize=V2_SS)
    declare_profile(m.v1, m.v2, wrt=m.t, profile="piecewise_constant")

    @m.Constraint(m.t)
    def zc_ode(m, t):
        return m.dzc[t] == (1 - m.zc[t]) / (m.u2sf * m.v2[t]) - m.k0 * m.zc[t] * pyo.exp(-m.ea / m.zt[t])

    @m.Constraint(m.t)
    def zt_ode(m, t):
        return m.dzt[t] == (m.ztf - m.zt[t]) / (m.u2sf * m.v2[t]) + m.k0 * m.zc[t] * pyo.exp(-m.ea / m.zt[t]) - m.a0 * m.u1sf * m.v1[t] * (m.zt[t] - m.ztcw)

    @m.Constraint()
    def zc_init(m):
        return m.zc[0] == m.zc0

    @m.Constraint()
    def zt_init(m):
        return m.zt[0] == m.zt0

    grid = sorted(m.t)

    # Tracking stage cost at the samples plus a state-only terminal cost
    # that pins the endpoint (suppresses the end-of-horizon drift a finite
    # horizon otherwise allows). Decorator form: an expr= objective cannot
    # be cloned by estimate().
    @m.Objective()
    def obj(m):
        return sum(
            10 * (m.zc[t] - m.zc_ss) ** 2 + 2 * (m.zt[t] - m.zt_ss) ** 2
            + (m.v1[t] - m.v1_ss) ** 2 + 0.5 * (m.v2[t] - m.v2_ss) ** 2
            for t in grid[:-1]) + 1000 * (10 * (m.zc[grid[-1]] - m.zc_ss) ** 2 + 2 * (m.zt[grid[-1]] - m.zt_ss) ** 2)

    pyo.TransformationFactory("dae.collocation").apply_to(
        m, wrt=m.t, nfe=N, ncp=3, scheme="LAGRANGE-RADAU")
    pyo.TransformationFactory("cvp.parameterize").apply_to(m)
    declare_sens_param(m.zc0, m.zt0)
    return m


def extract(m, ts):
    """Trajectories as plain lists. After cvp.parameterize only the move
    variables survive, one per sampling interval, so the control governing
    time t is the member at its element start."""
    return {
        "zc": [pyo.value(m.zc[t]) for t in ts],
        "zt": [pyo.value(m.zt[t]) for t in ts],
        "v1": [pyo.value(m.v1[min(int(t), N - 1)]) for t in ts],
        "v2": [pyo.value(m.v2[min(int(t), N - 1)]) for t in ts],
    }


def log_event(title, body=None):
    """Append to the session event log: a one-liner, or a title plus a
    captured solver log. Entries are numbered by a running per-session
    counter (numbers survive trimming), and the log keeps the most
    recent 50."""
    n = st.session_state.get("event_count", 0) + 1
    st.session_state["event_count"] = n
    events = st.session_state.setdefault("events", [])
    events.append({"n": n, "stamp": time.strftime("%H:%M:%S"),
                   "title": title, "body": body})
    if len(events) > 50:
        del events[:len(events) - 50]


# ── pounce worker thread ─────────────────────────────────────────────────────
# pounce keeps the KKT factorization inside a Rust solver object that is
# thread-affine: only the thread that created it may use it (or drop it).
# Streamlit runs every rerun on a fresh script thread, so a factorization
# created by the Solve click would be unusable by the Estimate click. All
# pounce work therefore runs on this single long-lived worker thread, which
# owns every solved model; script threads exchange plain data with it
# through queues.

class _PounceWorker:
    MAX_MODELS = 16  # baselines kept alive; the oldest evicted, on-thread

    def __init__(self):
        self._q = queue.Queue()
        self._store = {}  # token -> solved model, touched only on the thread
        self._order = []
        self._count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            fn, args, out = self._q.get()
            try:
                out.put((True, fn(self, *args)))
            except BaseException as e:  # a solver panic must not kill the thread
                out.put((False, e))

    def call(self, fn, *args, timeout=None):
        out = queue.Queue()
        self._q.put((fn, args, out))
        ok, val = out.get(timeout=timeout)
        if not ok:
            raise val
        return val

    def keep(self, m):
        """Store a solved model, evicting the oldest beyond MAX_MODELS.
        Worker-thread only. A Pyomo model is a web of reference cycles, so
        an evicted model waits for the cyclic GC; forcing that collection
        here keeps the solver drop on its owning thread."""
        self._count += 1
        self._store[self._count] = m
        self._order.append(self._count)
        evicted = False
        while len(self._order) > self.MAX_MODELS:
            del self._store[self._order.pop(0)]
            evicted = True
        if evicted:
            gc.collect()
        return self._count

    @staticmethod
    def _clear(w):
        w._store.clear()
        w._order.clear()
        gc.collect()

    def shutdown(self):
        """Best-effort store cleanup at interpreter shutdown: without it
        the stored models are dropped from the main thread, tripping the
        solver's thread-affinity check once per model in the server log."""
        try:
            self.call(_PounceWorker._clear, timeout=10)
        except Exception:
            pass


@st.cache_resource
def _worker():
    w = _PounceWorker()
    atexit.register(w.shutdown)
    return w


class _SolveLogCapture(io.TextIOBase):
    """Capture stdout during a worker-thread solve. pounce tees the
    engine's fd-level output to sys.stdout through a tail thread of its
    own, so the capture cannot key on the worker thread alone: it takes
    writes from every thread EXCEPT Streamlit script threads, whose
    output (say, a rich-colorized traceback from the exception logger)
    belongs on the terminal, not inside a solver log."""

    def __init__(self, real):
        self.real = real
        self.buf = io.StringIO()

    def write(self, s):
        if threading.current_thread().name.startswith("ScriptRunner"):
            self.real.write(s)
        else:
            self.buf.write(s)
        return len(s)

    def flush(self):
        self.real.flush()


@contextlib.contextmanager
def _capture_stdout():
    cap = _SolveLogCapture(sys.stdout)
    sys.stdout = cap
    try:
        yield cap.buf
    finally:
        sys.stdout = cap.real


def _solve_job(w, zc0, zt0):
    """Worker-thread job: build and solve the OCP, then read the gain
    matrix out of the held factorization: the first control move
    differentiated with respect to the initial state, one backsolve per
    entry, no extra solve."""
    m = build_model(zc0, zt0)
    t0 = time.perf_counter()
    with _capture_stdout() as buf:
        result = pyo.SolverFactory("pounce").solve(m, tee=True)
    solve_s = time.perf_counter() - t0
    status = str(result.solver.termination_condition)

    # A failed solve retains no factorization: no gains to read.
    K, gain_s = None, 0.0
    if status == "optimal":
        t0 = time.perf_counter()
        K = [[gradient(v[0], wrt=p) for p in (m.zc0, m.zt0)]
             for v in (m.v1, m.v2)]
        gain_s = time.perf_counter() - t0

    ts = sorted(m.t)
    samples = [t for t in ts if t == int(t)]
    return {
        "inputs": (zc0, zt0),
        "token": w.keep(m),
        "status": status,
        "log": buf.getvalue(),
        "solve_s": solve_s,
        "gain_s": gain_s,
        "K": K,
        "ts": [float(t) for t in ts],
        "sample_idx": [ts.index(t) for t in samples],
        "traj": extract(m, ts),
    }


def _perturb_job(w, token, zc0p, zt0p):
    """Worker-thread job: estimate() the re-optimized solution at the
    perturbed start from the baseline factorization, then re-solve exactly
    on a clone. The clone keeps the factorization with the baseline model,
    so perturbation after perturbation measures against the same
    reference."""
    if token not in w._store:
        raise LookupError("baseline expired")
    m = w._store[token]
    ts = sorted(m.t)

    t0 = time.perf_counter()
    est = estimate(m, [(m.zc0, zc0p), (m.zt0, zt0p)])
    est_s = time.perf_counter() - t0
    pred = {
        "zc": [est[m.zc[t]] for t in ts],
        "zt": [est[m.zt[t]] for t in ts],
        "v1": [est[m.v1[min(int(t), N - 1)]] for t in ts],
        "v2": [est[m.v2[min(int(t), N - 1)]] for t in ts],
    }

    m2 = m.clone()
    m2.zc0 = zc0p
    m2.zt0 = zt0p
    t0 = time.perf_counter()
    with _capture_stdout() as buf:
        result = pyo.SolverFactory("pounce").solve(m2, tee=True)
    resolve_s = time.perf_counter() - t0

    out = {
        "pert_inputs": (zc0p, zt0p),
        "status": str(result.solver.termination_condition),
        "log": buf.getvalue(),
        "est_s": est_s,
        "resolve_s": resolve_s,
        "pred": pred,
        "pert": extract(m2, ts),
        "moves": {
            "est": (est[m.v1[0]], est[m.v2[0]]),
            "exact": (pyo.value(m2.v1[0]), pyo.value(m2.v2[0])),
        },
    }
    # The clone's reference cycles would otherwise reach the cyclic GC
    # from an arbitrary script thread; its solver must drop here.
    del m2, est
    gc.collect()
    return out


def solve_baseline(zc0, zt0):
    """Solve on the worker thread, then record the events. The returned
    dict holds plain data plus the worker's token for the stored model.
    Solves are numbered per session, and every event names the solve it
    belongs to."""
    res = _worker().call(_solve_job, zc0, zt0)
    log_event(f"solve from zc(0) = {zc0:.2f}, zt(0) = {zt0:.2f}: "
              f"{res['status']} in {res['solve_s']:.2f} s", res.pop("log"))
    if res["K"] is not None:
        log_event(f"gain matrix: 4 backsolves against the held "
                  f"factorization in {res['gain_s'] * 1e6:.0f} microseconds")
    return res


def run_perturbation(base, zc0p, zt0p):
    """Estimate and re-solve on the worker thread, then record the events.
    Raises LookupError when the stored baseline has been evicted."""
    res = _worker().call(_perturb_job, base["token"], zc0p, zt0p)
    log_event(f"estimate at zc(0) = {zc0p:.2f}, zt(0) = {zt0p:.2f}: "
              f"full-solution prediction in {res['est_s'] * 1e6:.0f} "
              f"microseconds, no solver run")
    log_event(f"re-solve from zc(0) = {zc0p:.2f}, zt(0) = {zt0p:.2f}: "
              f"{res['status']} in {res['resolve_s']:.2f} s", res.pop("log"))
    res["base_inputs"] = base["inputs"]
    return res


# ── Sidebar control fragments ────────────────────────────────────────────────
# Each section is a fragment: moving its sliders reruns only that section,
# not the whole page, while its button's disable logic stays live. Since a
# fragment cannot see the other section's slider moves, both buttons key
# off the STORED baseline, never the other fragment's slider positions:
# the displayed results always belong to the stored baseline, so that is
# the reference that matters. A click escalates to a full-app rerun.
#
# Slider ranges: zc(0) spans the state's variable bounds [0, 1]. zt has
# no upper variable bound, so the cap sits above the operating region.
# The floor 0.52 keeps every start settling at the steady state within
# the horizon for every zc(0) (swept at N = 50; N = 100 would allow
# 0.49 but no colder: below that the Arrhenius term is essentially
# frozen, the coolant can only remove heat, and ignition outlasts any
# reasonable horizon; the empty reactor zc(0) = 0 is the binding case).

@st.fragment
def _ic_controls():
    st.markdown("## Initial Condition")
    zc0 = st.slider("$z_c$ concentration", 0.0, 1.0, 0.0, 0.01,
                    format="%.2f", key="zc0")
    zt0 = st.slider("$z_t$ temperature", 0.52, 0.70, 0.52, 0.01,
                    format="%.2f", key="zt0")
    base = st.session_state.get("base")
    solved = base is not None and base["inputs"] == (zc0, zt0)
    clicked = st.button("Solve", type="primary", use_container_width=True,
                        disabled=solved)
    if clicked:
        try:
            res = solve_baseline(zc0, zt0)
        except Exception as e:
            st.error(f"Solver error: {e}")
            st.stop()
        st.session_state["base"] = res
        # A new baseline invalidates any existing comparison: the
        # estimate was taken from the old factorization.
        st.session_state.pop("cmp", None)
        st.session_state["solve_status"] = res["status"]
        st.rerun(scope="app")


@st.fragment
def _perturb_controls():
    st.markdown("## Perturbed Start")
    zc0p = st.slider("$z_c$ concentration", 0.0, 1.0, 0.61, 0.01,
                     format="%.2f", key="zc0p")
    zt0p = st.slider("$z_t$ temperature", 0.52, 0.70, 0.54, 0.01,
                     format="%.2f", key="zt0p")
    base = st.session_state.get("base")
    cmp_res = st.session_state.get("cmp")
    no_factorization = base is None or base["K"] is None
    cmp_current = (not no_factorization and cmp_res is not None
                   and cmp_res["pert_inputs"] == (zc0p, zt0p)
                   and cmp_res["base_inputs"] == base["inputs"])
    clicked = st.button("Estimate then Re-solve", type="primary",
                        use_container_width=True,
                        disabled=no_factorization or cmp_current)
    if clicked:
        try:
            res = run_perturbation(base, zc0p, zt0p)
        except LookupError:
            # The stored baseline was evicted from the worker's cache:
            # drop the stale results and ask for a fresh solve.
            st.session_state.pop("base", None)
            st.session_state.pop("cmp", None)
            st.session_state["expired"] = True
            st.rerun(scope="app")
        except Exception as e:
            st.error(f"Solver error: {e}")
            st.stop()
        st.session_state["cmp"] = res
        st.session_state["solve_status"] = res["status"]
        st.rerun(scope="app")


# The sidebar is assembled further down, after the figure builders exist:
# the gains chart renders between the two control fragments.


# ── Figures ──────────────────────────────────────────────────────────────────

def show_schematic():
    """The static CSTR schematic, pre-rendered to schematic.png (the same
    image the notebook embeds) so the app carries no drawing code. Shown
    only on the Formulation tab."""
    st.image(str(Path(__file__).parent / "schematic.png"), width="stretch")


def build_gain_chart(base):
    """The four local gains as sign-colored horizontal bars, sized for the
    sidebar: green rightward for a positive gain, red leftward for a
    negative one, the value at each bar tip. Horizontal because at
    sidebar width the four partial-derivative labels only fit as y-tick
    rows. With no factorization (base None) the same frame renders with
    no bars, so the sidebar layout is identical before the first solve."""
    labels = [r"$\partial v_1 / \partial z_c$",
              r"$\partial v_1 / \partial z_t$",
              r"$\partial v_2 / \partial z_c$",
              r"$\partial v_2 / \partial z_t$"]
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    # Transparent background: the chart sits on the sidebar's own shade.
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    if base is not None:
        K = base["K"]
        vals = [K[0][0], K[0][1], K[1][0], K[1][1]]
        colors = ["#009E73" if v >= 0 else "#CC3311" for v in vals]
        ax.barh(range(4), vals, color=colors, height=0.55)
        for k, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (v, k),
                        xytext=(4 if v >= 0 else -4, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8)
        lo = min(vals + [0.0])
        hi = max(vals + [0.0])
        rng = hi - lo if hi > lo else 1.0
        # A side with a bar carries the value annotation beyond the bar
        # tip: pad that side generously so the text never overlaps the
        # tick labels or clips at the frame.
        ax.set_xlim(lo - (0.32 if lo < 0 else 0.05) * rng,
                    hi + (0.32 if hi > 0 else 0.05) * rng)
    else:
        ax.set_xlim(-1.0, 1.0)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # matrix row order, top to bottom
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _foreground(*axes, unclip=True):
    """Draw the data above the axes frame. Spines render at a higher
    z-order than lines, so a trajectory sitting exactly on an axis bound
    would otherwise hide under the frame; unclipping additionally keeps
    the full line width visible at the edge. The phase plots keep
    clipping (unclip=False): their temperature window is the realistic
    trajectory envelope (swept extremes 0.458 and 0.816 over the whole
    slider box), and an estimate extrapolating past it must not spill
    outside the frame."""
    for ax in axes:
        for artist in ax.lines:
            artist.set_zorder(3)
            if unclip:
                artist.set_clip_on(False)


def build_timeseries(base):
    """Baseline optimal trajectories: states over controls, targets
    dashed. All axes fixed to [0, 1]: every plotted quantity lives
    there, and the figures never rescale."""
    ts, traj = base["ts"], base["traj"]
    fig, (ax_z, ax_v) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    ax_z.plot(ts, traj["zc"], color=WONG[0], label="$z_c$ concentration")
    ax_z.plot(ts, traj["zt"], color=WONG[1], label="$z_t$ temperature")
    ax_z.axhline(ZC_SS, color=WONG[0], ls="--", lw=1, alpha=0.6)
    ax_z.axhline(ZT_SS, color=WONG[1], ls="--", lw=1, alpha=0.6)
    ax_z.set_ylabel("states")
    ax_z.legend(loc="lower right")
    ax_v.step(ts, traj["v1"], where="post", color=WONG[2],
              label="$v_1$ coolant flow")
    ax_v.step(ts, traj["v2"], where="post", color=WONG[3],
              label="$v_2$ residence time")
    ax_v.axhline(V1_SS, color=WONG[2], ls="--", lw=1, alpha=0.6)
    ax_v.axhline(V2_SS, color=WONG[3], ls="--", lw=1, alpha=0.6)
    ax_v.set_ylabel("controls")
    ax_v.set_xlabel("time")
    ax_v.legend(loc="lower right")
    ax_z.set_ylim(0.0, 1.0)
    ax_v.set_ylim(0.0, 1.0)
    _foreground(ax_z, ax_v)
    fig.subplots_adjust(left=0.078, right=0.97, top=0.97, bottom=0.10, hspace=0.10)
    return fig


def build_phase(base):
    """Baseline trajectory in the phase plane: initial condition (dot) into
    the steady-state target (cross), dots at the sample times."""
    ts, traj, idx = base["ts"], base["traj"], base["sample_idx"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(traj["zc"], traj["zt"], color=WONG[0], label="state trajectory")
    ax.plot([traj["zc"][k] for k in idx], [traj["zt"][k] for k in idx],
            ".", color=WONG[0], ms=6, alpha=0.7)
    ax.plot(*base["inputs"], "o", color=WONG[0], ms=9)
    ax.plot(ZC_SS, ZT_SS, "X", color="k", ms=10, label="steady-state target")
    ax.set_xlabel("$z_c$ concentration")
    ax.set_ylabel("$z_t$ temperature")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.45, 0.85)
    _foreground(ax, unclip=False)
    fig.subplots_adjust(left=0.118, right=0.97, top=0.97, bottom=0.12)
    return fig


def build_comparison_timeseries(base, cmp_res):
    """Perturbed start in time: the sensitivity estimate (dashed) against
    the exact re-solve (solid), states over controls, targets dotted."""
    ts = base["ts"]
    pert, pred = cmp_res["pert"], cmp_res["pred"]
    fig, (ax_z, ax_v) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    ax_z.plot(ts, pert["zc"], color=WONG[0], label="$z_c$ re-solve")
    ax_z.plot(ts, pert["zt"], color=WONG[1], label="$z_t$ re-solve")
    ax_z.plot(ts, pred["zc"], color=WONG[0], ls="--", label="$z_c$ estimate")
    ax_z.plot(ts, pred["zt"], color=WONG[1], ls="--", label="$z_t$ estimate")
    ax_z.axhline(ZC_SS, color=WONG[0], ls=":", lw=1, alpha=0.6)
    ax_z.axhline(ZT_SS, color=WONG[1], ls=":", lw=1, alpha=0.6)
    ax_z.set_ylabel("states")
    ax_z.legend(loc="lower right", fontsize=8)
    ax_v.step(ts, pert["v1"], where="post", color=WONG[2],
              label="$v_1$ re-solve")
    ax_v.step(ts, pert["v2"], where="post", color=WONG[3],
              label="$v_2$ re-solve")
    ax_v.step(ts, pred["v1"], where="post", color=WONG[2], ls="--",
              label="$v_1$ estimate")
    ax_v.step(ts, pred["v2"], where="post", color=WONG[3], ls="--",
              label="$v_2$ estimate")
    ax_v.axhline(V1_SS, color=WONG[2], ls=":", lw=1, alpha=0.6)
    ax_v.axhline(V2_SS, color=WONG[3], ls=":", lw=1, alpha=0.6)
    ax_v.set_ylabel("controls")
    ax_v.set_xlabel("time")
    ax_v.legend(loc="lower right", fontsize=8)
    ax_z.set_ylim(0.0, 1.0)
    ax_v.set_ylim(0.0, 1.0)
    _foreground(ax_z, ax_v)
    fig.subplots_adjust(left=0.078, right=0.97, top=0.97, bottom=0.10, hspace=0.10)
    return fig


def build_phase_pert(base, cmp_res):
    """Perturbed start in the phase plane: the exact re-solve (solid, dots
    at the sample times) against the sensitivity estimate (dashed)."""
    idx = base["sample_idx"]
    pert, pred = cmp_res["pert"], cmp_res["pred"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(pert["zc"], pert["zt"], color=WONG[0], label="re-solve")
    ax.plot([pert["zc"][k] for k in idx], [pert["zt"][k] for k in idx],
            ".", color=WONG[0], ms=6, alpha=0.7)
    ax.plot(pred["zc"], pred["zt"], color=WONG[2], ls="--",
            label="sensitivity estimate")
    ax.plot(*cmp_res["pert_inputs"], "o", color=WONG[0], ms=9)
    ax.plot(ZC_SS, ZT_SS, "X", color="k", ms=10, label="steady-state target")
    ax.set_xlabel("$z_c$ concentration")
    ax.set_ylabel("$z_t$ temperature")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.45, 0.85)
    _foreground(ax, unclip=False)
    fig.subplots_adjust(left=0.118, right=0.97, top=0.97, bottom=0.12)
    return fig


# matplotlib's shared state (the mathtext parser especially) is not
# thread-safe, and Streamlit can interrupt a rerun mid-render while the
# next script thread draws: without a lock, concurrent "$z_c$" label
# parses corrupt each other and raise ParseException. One lock covers
# figure building and rendering together.
_MPL_LOCK = threading.RLock()


def show(builder, *args, fixed_frame=False):
    # No explicit width argument on st.pyplot: the default stretches to
    # the container, and the legacy use_container_width= is deprecated.
    # fixed_frame skips Streamlit's tight bounding-box trim: the paired
    # figures use fixed subplot margins instead, so their frames stay
    # pixel-stationary when the shared axis limits change (a tight trim
    # would shift with every tick-label width).
    with _MPL_LOCK:
        fig = builder(*args)
        if fixed_frame:
            st.pyplot(fig, bbox_inches=None)
        else:
            st.pyplot(fig)
        plt.close(fig)


# ── Sidebar assembly ─────────────────────────────────────────────────────────
# The gains section sits between the two control fragments and outside
# both: it refreshes exactly when a Solve completes (a full-app rerun),
# and slider drags never touch it. Before an optimal solve holds a
# factorization the chart renders as an empty frame, so the sidebar
# layout never shifts.

with st.sidebar:
    _ic_controls()
    st.markdown("## Local Feedback Gains")
    _base = st.session_state.get("base")
    # Fixed-height box reserves the chart's space before the image
    # arrives, so the Perturbed Start section below never moves while
    # the figure renders.
    with st.container(height=190, border=False):
        show(build_gain_chart,
             _base if (_base is not None and _base["K"] is not None) else None)
    _perturb_controls()


# ── Formulation tab ──────────────────────────────────────────────────────────

def render_formulation_tab():
    col_text, col_fig = st.columns([5, 4])
    with col_text:
        st.markdown(r"""
### The reactor

The Hicks-Ray CSTR [1] runs an exothermic first-order reaction
A $\rightarrow$ B in a cooled continuous stirred-tank reactor. The app uses
the dimensionless form of Huang, Patwardhan and Biegler [2]: two states,
concentration $z_c$ and temperature $z_t$, and two manipulated inputs,
coolant flow $u_1$ and residence time $u_2$:

$$\dot{z}_c = \frac{1 - z_c}{u_2} - k_0\, z_c\, e^{-E_a / z_t}$$

$$\dot{z}_t = \frac{z_{t,f} - z_t}{u_2} + k_0\, z_c\, e^{-E_a / z_t} - a_0\, u_1 (z_t - z_{t,cw})$$

The solver works in scaled controls $v_1 = u_1 / 600 \in [1/6,\, 1]$ and
$v_2 = u_2 / 40 \in [0.025,\, 1]$. The regulation target
$(z_c^{ss}, z_t^{ss}) = (0.6416,\, 0.5387)$ is the model's **open-loop
unstable** steady state: without feedback the reactor drifts away from it,
which is what makes the problem a control benchmark.
""")
    with col_fig:
        show_schematic()

    st.markdown("""
<div style="font-size: 0.95rem; color: #495057; margin: 0.25rem 0 1.25rem 0;">
k<sub>0</sub> = 300 &nbsp;·&nbsp; E<sub>a</sub> = 5 &nbsp;·&nbsp;
a<sub>0</sub> = 1.95×10<sup>-4</sup> &nbsp;·&nbsp;
z<sub>t,f</sub> = 0.395 &nbsp;·&nbsp; z<sub>t,cw</sub> = 0.38 &nbsp;·&nbsp;
v<sub>1</sub><sup>ss</sup> = 0.57828 &nbsp;·&nbsp;
v<sub>2</sub><sup>ss</sup> = 0.49989
</div>
""", unsafe_allow_html=True)

    st.markdown(r"""
### The optimal control problem

An NMPC controller computes its action by solving an optimal control
problem from the plant's current state. Here that problem is direct
collocation (Radau, 3 points per element) over a horizon of $N = 100$
one-unit sampling intervals, with piecewise-constant controls, one move
per interval, declared with pyomo-cvp before discretization and
parameterized after, so each control has exactly one decision variable per
move. The objective is a tracking stage cost toward the steady state plus
a state-only terminal cost that pins the endpoint:

$$\min \;\; \sum_{k=0}^{N-1} \Big( 10\, (z_{c,k} - z_c^{ss})^2 + 2\, (z_{t,k} - z_t^{ss})^2 + (v_{1,k} - v_1^{ss})^2 + \tfrac{1}{2} (v_{2,k} - v_2^{ss})^2 \Big) \;+\; 1000 \Big( 10\, (z_{c,N} - z_c^{ss})^2 + 2\, (z_{t,N} - z_t^{ss})^2 \Big)$$

subject to the collocation equations of the two ODEs, the control bounds,
and the initial condition $z_c(0) = z_{c0}$, $z_t(0) = z_{t0}$. The initial
condition enters as two **mutable parameters** flagged with
`declare_sens_param`: everything downstream differentiates with respect
to them.

### Sensitivity: gains and estimates from one factorization

The solver that just found the optimum is still holding the KKT
factorization from its last iteration. sIPOPT-style parametric sensitivity
[4] reads two things out of it without any further solver run:

- **The local gain matrix.** Each entry of
  $K = \partial (v_{1,0},\, v_{2,0}) \, / \, \partial (z_{c0},\, z_{t0})$
  is one backsolve: the derivative of a first control move with respect to
  one component of the initial state. This is the local feedback law
  around the solved trajectory.
- **The full re-optimized solution at a perturbed start.** `estimate()`
  applies the same backsolve to every variable at once, a first-order
  Taylor step of the entire solution in the initial condition. This is
  the heart of advanced-step NMPC [3]: the estimate is excellent near the
  solved point and degrades as the perturbation grows, and the app lets
  you push it until it breaks. The exact re-solve runs on a copy of the
  model, so the factorization stays with the baseline and every estimate
  is measured against the same reference.

### Solution method

Built with [Pyomo](https://github.com/Pyomo/pyomo) and pyomo.dae;
piecewise-constant moves via
[pyomo-cvp](https://pypi.org/project/pyomo-cvp/); solved with POUNCE, a
Rust reimplementation of the IPOPT primal-dual interior-point algorithm,
whose [pyomo-pounce](https://pypi.org/project/pyomo-pounce/) wheel also
provides `declare_sens_param`, `gradient`, and `estimate`.

See the [companion Jupyter notebook](https://github.com/devin-griff/cstr-sensitivity/blob/main/CSTR%20sensitivity.ipynb)
for the full implementation this app is built from.

### References

[1] G. A. Hicks and W. H. Ray, "Approximation methods for optimal control
synthesis," *Can. J. Chem. Eng.*, vol. 49, pp. 522-528, 1971.
[DOI](https://doi.org/10.1002/cjce.5450490416)

[2] R. Huang, S. C. Patwardhan, and L. T. Biegler, "Robust stability of
nonlinear model predictive control based on extended Kalman filter,"
*J. Process Control*, vol. 22, pp. 82-89, 2012.
[DOI](https://doi.org/10.1016/j.jprocont.2011.10.006)

[3] V. M. Zavala and L. T. Biegler, "The advanced-step NMPC controller:
optimality, stability and robustness," *Automatica*, vol. 45, pp. 86-93,
2009.
[DOI](https://doi.org/10.1016/j.automatica.2008.06.011)

[4] H. Pirnay, R. Lopez-Negrete, and L. T. Biegler, "Optimal sensitivity
based on IPOPT," *Math. Program. Comput.*, vol. 4, pp. 307-331, 2012.
[DOI](https://doi.org/10.1007/s12532-012-0043-2)
""")


# ── Main layout ──────────────────────────────────────────────────────────────
# The button clicks are handled inside the sidebar fragments above; this
# section renders the page against whatever they stored.

if st.session_state.pop("expired", False):
    st.toast("The cached baseline solve expired: click Solve to recompute "
             "it.", icon="⚠️")

# Warn once per solve when the solver returns a non-optimal status; the
# happy path stays silent.
_status = st.session_state.pop("solve_status", None)
if _status is not None and _status != "optimal":
    st.toast(f"Solver status: {_status}: results may be inaccurate.",
             icon="⚠️")

st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; "
    "font-weight: 700;'>"
    "CSTR Sensitivity: Local Feedback from One Solve "
    "<a href='https://github.com/devin-griff/cstr-sensitivity' target='_blank' "
    "title='View source on GitHub' "
    "style='display: inline-block; vertical-align: 0.02em; "
    "margin: 0 0.35rem 0 0.1rem; color: inherit;'>"
    "<svg viewBox='0 0 16 16' width='20' height='20' fill='currentColor' "
    "aria-label='GitHub'>"
    "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17."
    "55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
    ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
    "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59."
    "82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
    ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51"
    ".56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1."
    "07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-"
    "8-8-8z'/></svg></a>"
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/jkitchin/pounce' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>POUNCE</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([6, 3])
with _caption_col:
    st.markdown(
        "Solve the CSTR's optimal control problem from an initial "
        "condition. Local feedback gains come automatically from "
        "pyomo-pounce. Then, perturb the initial condition to see the "
        "estimated trajectory as well as the resolved trajectory. "
        "**Time Series** and **Phase Plot** show the results, "
        "**Formulation** explains the model, and **Logs** keeps the "
        "session's event log."
    )

tab_ts, tab_ph, tab_form, tab_logs = st.tabs(
    ["📈  Time Series", "🌀  Phase Plot", "📐  Formulation", "📋  Logs"])

base = st.session_state.get("base")
cmp_res = st.session_state.get("cmp")
# The comparison only renders against the baseline it was computed from;
# a new solve clears it, so a mismatch just means "not estimated yet".
cmp_ok = (base is not None and cmp_res is not None
          and cmp_res["base_inputs"] == base["inputs"])


def _prompt_estimate():
    if base["K"] is None:
        st.info("No held factorization to estimate from.")
    else:
        st.info("Choose a perturbed start in the sidebar and click "
                "**Estimate then Re-solve**: the plant never starts "
                "exactly where the last solve assumed.")


with tab_ts:
    if base is None:
        col_text, _ = st.columns([6, 3])
        with col_text:
            st.info("Set the initial condition in the sidebar and click "
                    "**Solve**.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Baseline")
            show(build_timeseries, base, fixed_frame=True)
        with col_r:
            st.markdown("#### Perturbed start")
            if not cmp_ok:
                _prompt_estimate()
            else:
                show(build_comparison_timeseries, base, cmp_res,
                     fixed_frame=True)

with tab_ph:
    if base is None:
        col_text, _ = st.columns([6, 3])
        with col_text:
            st.info("Set the initial condition in the sidebar and click "
                    "**Solve**.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Baseline")
            show(build_phase, base, fixed_frame=True)
        with col_r:
            st.markdown("#### Perturbed start")
            if not cmp_ok:
                _prompt_estimate()
            else:
                show(build_phase_pert, base, cmp_res, fixed_frame=True)

with tab_form:
    render_formulation_tab()

with tab_logs:
    events = st.session_state.get("events", [])
    if not events:
        st.info("Click **Solve** to start the event log: full POUNCE output "
                "for every solve, one-liners for the factorization "
                "backsolves.")
    else:
        # Chronological, in a fixed-height scroll box pinned to the
        # newest entry. The pinning JS below scrolls the box on render
        # (when Logs is the active tab) and on every later click of the
        # Logs tab button, since tab switches happen client-side with no
        # rerun. Only height-limited block containers are touched, never
        # the page scroller.
        with st.container(height=560, border=False):
            for ev in events:
                st.markdown(f"**{ev['n']} &nbsp;·&nbsp; {ev['stamp']} "
                            f"&nbsp; {ev['title']}**")
                if ev["body"]:
                    st.code(ev["body"], language=None)
        st.iframe(f"""
<script>
(function() {{
    // re-render key (forces this script to run per new entry): {len(events)}-{events[-1]['n']}
    const doc = window.parent.document;
    // The height-limited container's scrolling element is a nested
    // stVerticalBlock with overflow-y auto (verified in the live DOM);
    // it is the only such element under stMain.
    const scrollBottom = () => {{
        const sel = '[data-testid="stMain"] [data-testid="stVerticalBlock"]';
        for (const el of doc.querySelectorAll(sel)) {{
            const oy = getComputedStyle(el).overflowY;
            if ((oy === 'auto' || oy === 'scroll') && el.clientHeight > 0
                    && el.scrollHeight > el.clientHeight) {{
                el.scrollTop = el.scrollHeight;
            }}
        }}
    }};
    let tries = 0;
    const tick = () => {{ scrollBottom(); if (++tries < 8) setTimeout(tick, 150); }};
    tick();
    if (!window.parent.__logsScrollWired) {{
        window.parent.__logsScrollWired = true;
        // Streamlit tabs are div[role="tab"], not buttons.
        doc.addEventListener('click', (e) => {{
            const t = e.target.closest && e.target.closest('[role="tab"]');
            if (t && t.textContent.includes('Logs')) setTimeout(scrollBottom, 80);
        }}, true);
    }}
}})();
</script>
""", height=1)
