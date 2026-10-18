#!/usr/bin/env python3
"""
HUA QoS-aware AI inference offloading — live demonstrator — IEEE CloudNet 2026.

Runs the full closed loop in one terminal window and renders it as a live
dashboard: a simulated vehicle drives between MEC cells, the ML predictor
estimates network conditions per frame, and the QoS optimizer picks one of
four YOLOv8 execution tiers under hard latency / accuracy / capacity budgets.

    A  on-board NPU   yolov8n    lowest latency, worst accuracy
    B  on-board GPU   yolov8n    emergency fallback
    C  MEC edge       yolov8m    preferred tier (edge-first rule)
    D  cloud          yolov8l    best accuracy, highest RTT

Everything is simulated from the bundled physics-based dataset — no network,
no GPU, and no external hardware are required.

Live controls (any key, no Enter needed)::

    1-5      switch road scenario          o   inject MEC outage      (C -> D)
    space    pause / resume                x   inject full blackout   (C -> B)
    + / -    faster / slower               c   inject cloud congestion
    k        toggle kiosk auto-pilot       r   reset session counters
    q        quit

Usage::

    python demo.py                              # run until 'q', outage at frame 40
    python demo.py --frames 100                 # finite paper run
    python demo.py --kiosk                      # unattended booth loop (recommended)
    python demo.py --scenario highway --hz 10
    python demo.py --frames 0                   # run until 'q'
    python demo.py --quantile                   # risk-aware q90 gating
    python demo.py --outage-at 0                # no injected outage
"""

from __future__ import annotations

import argparse
import re
import select
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:                                    # POSIX only; the demo degrades gracefully
    import termios
    import tty
except ImportError:                     # pragma: no cover - Windows
    termios = tty = None                # type: ignore[assignment]

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import (
    ACC_LOSS_BUDGET,
    CAPACITY_LIMIT_PCT,
    DATA_PATH,
    LATENCY_BUDGET_MS,
    MEC_PREFERENCE_MAP_DELTA,
    MODEL_PATH,
    QUANTILE_PATH,
    SCENARIOS,
)
from src.dataset.generate import generate_dataset
from src.offloading.optimizer import DwellTimeStabilizer, explain, select_mode
from src.offloading.table import PerformanceTable
from src.predictor.client import InferenceClient
from src.predictor.model import train

# ── presentation constants ───────────────────────────────────────────────────

MODE_STYLE = {"A": "yellow", "B": "bright_cyan", "C": "bright_green", "D": "bright_magenta"}
MODE_TIER  = {"A": "on-board NPU", "B": "on-board GPU", "C": "MEC edge", "D": "cloud DC"}

SCENARIO_ORDER = ["urban_dense", "urban_sparse", "suburban", "highway", "rural"]
SCENARIO_BLURB = {
    "urban_dense":  "dense city grid, 300 m cells, 100 MHz @ 3.5 GHz, heavy UE contention",
    "urban_sparse": "city outskirts, 800 m cells, 100 MHz @ 3.5 GHz",
    "suburban":     "suburban arterial, 2 km cells, 40 MHz @ 2.1 GHz",
    "highway":      "motorway, 5 km cells, 20 MHz @ 2.1 GHz, 80-130 km/h",
    "rural":        "rural road, 10 km cells, 10 MHz @ 900 MHz, frequent NLOS",
}

# Human-readable gloss for every reason string the optimizer can emit.
REASON_TEXT = {
    "edge_first_within_map_delta":
        f"edge-first: cloud buys < {MEC_PREFERENCE_MAP_DELTA:.0f} mAP pts, so keep the traffic at the MEC",
    "cloud_accuracy_gain_justifies_wan":
        f"cloud accuracy gain >= {MEC_PREFERENCE_MAP_DELTA:.0f} mAP pts, worth the WAN round-trip",
    "mec_unavailable_cloud_fallback":
        "MEC unreachable, cloud still inside every budget",
    "remote_tiers_infeasible_onboard_fallback":
        "both remote tiers infeasible, running on board",
    "only_feasible_mode":
        "the only tier inside every QoS budget",
    "best_feasible":
        "highest mAP among the feasible tiers",
    "fallback_no_feasible_mode":
        "NO feasible tier: accuracy budget yields so the vehicle keeps perceiving",
    "dwell_hold_previous_tier":
        "holding the current tier until the proposed switch survives the dwell window",
}

# 5x7 block glyphs so the selected tier is readable from across the room.
BIG_LETTER = {
    "A": [" █████ ", "██   ██", "███████", "██   ██", "██   ██"],
    "B": ["██████ ", "██   ██", "██████ ", "██   ██", "██████ "],
    "C": [" ██████", "██     ", "██     ", "██     ", " ██████"],
    "D": ["██████ ", "██   ██", "██   ██", "██   ██", "██████ "],
}

TAGLINES = [
    "Every frame, the optimizer re-checks three hard budgets: 100 ms end-to-end, 15% accuracy loss, 95% server capacity.",
    "Modes A and B run YOLOv8n on board: fast, but 20.8% less accurate than the cloud model, so they stay infeasible while the network holds.",
    "The edge-first rule keeps inference at the MEC whenever the cloud is not meaningfully more accurate, conserving WAN bandwidth.",
    "Press 'x' to cut the network: watch the system fall back to on-board GPU and then recover to the MEC.",
    "The whole channel model is physics-based and 3GPP TR 38.901-calibrated, so the latency spikes are earned, not scripted.",
]

# ── small rendering helpers ──────────────────────────────────────────────────

_INFEASIBLE_RE = {
    "latency":  re.compile(r"^latency_(\d+)ms_over_budget$"),
    "acc":      re.compile(r"^acc_loss_(\d+)pct_over_budget$"),
    "capacity": re.compile(r"^capacity_(\d+)pct_saturated$"),
}


def pretty_violation(reason: str) -> str:
    """Turn an optimizer constraint flag into something a visitor can read."""
    if reason == "mec_unreachable":
        return "MEC unreachable"
    if m := _INFEASIBLE_RE["latency"].match(reason):
        return f"latency {m.group(1)} > {LATENCY_BUDGET_MS:.0f} ms"
    if m := _INFEASIBLE_RE["acc"].match(reason):
        return f"loss {m.group(1)}% > {ACC_LOSS_BUDGET * 100:.0f}%"
    if m := _INFEASIBLE_RE["capacity"].match(reason):
        return f"cap {m.group(1)}% >= {CAPACITY_LIMIT_PCT:.0f}%"
    return reason.replace("_", " ")


def budget_bar(value: float | None, budget: float, width: int = 22) -> Text:
    """Horizontal gauge with a dotted marker at the QoS budget."""
    if value is None:
        return Text("·" * width, style="grey35")
    span   = budget * 1.6
    filled = int(round(min(value / span, 1.0) * width))
    mark   = min(int(round(budget / span * width)), width - 1)
    if value > budget:
        style = "bold red"
    elif value > budget * 0.75:
        style = "yellow"
    else:
        style = "green"
    out = Text()
    for i in range(width):
        if i < filled:
            out.append("█", style=style)
        elif i == mark:
            out.append("┊", style="bright_white")
        else:
            out.append("·", style="grey35")
    return out


def signal_bars(rssi: float) -> Text:
    """Five-bar signal meter from received power (-110 dBm .. -50 dBm)."""
    quality = max(0.0, min(1.0, (rssi + 110.0) / 60.0))
    lit     = int(round(quality * 5))
    style   = "bright_green" if lit >= 4 else "yellow" if lit >= 2 else "red"
    out     = Text()
    for i, glyph in enumerate("▁▃▅▆█"):
        out.append(glyph, style=style if i < lit else "grey30")
    return out


def road_view(dist_m: float, spacing_m: float, width: int, mec_up: bool) -> Text:
    """One-line map of the vehicle between the MEC cell and the cell edge."""
    width = max(20, width)
    frac  = max(0.0, min(1.0, dist_m / spacing_m))
    pos   = int(frac * (width - 1))
    link  = "bright_green" if mec_up else "red"
    out   = Text()
    out.append("▟▙ MEC ", style=f"bold {link}")
    for i in range(width):
        if i == pos:
            out.append("◆", style="bold bright_white")
        elif i < pos:
            out.append("─" if mec_up else "╌", style=link)
        else:
            out.append("·", style="grey30")
    out.append(" cell edge", style="grey50")
    return out


def big_mode(mode: str) -> Text:
    """Render the selected tier as block capitals."""
    out = Text()
    for i, row in enumerate(BIG_LETTER[mode]):
        out.append(row, style=f"bold {MODE_STYLE[mode]}")
        if i < len(BIG_LETTER[mode]) - 1:
            out.append("\n")
    return out


# ── session state ────────────────────────────────────────────────────────────

@dataclass
class Injection:
    """A fault the presenter (or the kiosk auto-pilot) forced into the link."""

    kind:        str
    frames_left: int

    LABEL = {
        "mec_outage":  "MEC OUTAGE",
        "blackout":    "FULL BLACKOUT",
        "cloud_spike": "CLOUD CONGESTION",
    }

    @property
    def label(self) -> str:
        return self.LABEL[self.kind]

    def apply(self, prediction: dict) -> dict:
        """Return a copy of the prediction with the fault applied."""
        p = dict(prediction)
        if self.kind in ("mec_outage", "blackout"):
            p["mec_available"] = 0
        if self.kind == "blackout":
            p["cloud_network_ms"] = 200.0          # 4 + 200 >> 100 ms budget
        elif self.kind == "cloud_spike":
            p["cloud_network_ms"] = max(p["cloud_network_ms"] * 2.4, 140.0)
        if "cloud_network_ms_q" in p:              # keep q90 gating in step
            p["cloud_network_ms_q"] = max(p["cloud_network_ms_q"], p["cloud_network_ms"])
        return p


@dataclass
class Stats:
    """Counters shown in the session panel and the closing summary."""

    frames:        int     = 0
    switches:      int     = 0
    deadline_miss: int     = 0
    acc_breach:    int     = 0
    mec_up:        int     = 0
    latency_sum:   float   = 0.0
    dwell:         Counter = field(default_factory=Counter)
    started:       float   = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self.frames = self.switches = self.deadline_miss = self.acc_breach = 0
        self.mec_up = 0
        self.latency_sum = 0.0
        self.dwell.clear()
        self.started = time.monotonic()

    @property
    def uptime(self) -> str:
        s = int(time.monotonic() - self.started)
        return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"

    @property
    def mean_latency(self) -> float:
        return self.latency_sum / self.frames if self.frames else 0.0

    @property
    def compliance(self) -> float:
        good = self.frames - self.deadline_miss - self.acc_breach
        return 100.0 * good / self.frames if self.frames else 100.0


class KeyReader:
    """Non-blocking single-keystroke reader (cbreak mode, restored on exit)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled and termios and sys.stdin.isatty())
        self._saved  = None

    def __enter__(self) -> "KeyReader":
        if self.enabled:
            self._fd    = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def get(self, timeout: float) -> str | None:
        """Return one keystroke, or None if nothing arrived within *timeout*."""
        if not self.enabled:
            time.sleep(max(0.0, timeout))
            return None
        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
        return sys.stdin.read(1) if ready else None


# ── panels ───────────────────────────────────────────────────────────────────

def header_panel(state: "Demo") -> Panel:
    bar = Table.grid(expand=True)
    bar.add_column(justify="left")
    bar.add_column(justify="center")
    bar.add_column(justify="right")

    left = Text()
    left.append("QoS-AWARE AI INFERENCE OFFLOADING", style="bold bright_white")
    left.append("  ·  vehicular edge computing", style="grey62")

    mid = Text()
    mid.append(state.scenario.replace("_", " "), style="bold bright_yellow")
    mid.append(f"  {SCENARIO_BLURB[state.scenario]}", style="grey58")

    right = Text()
    if state.paused:
        right.append(" PAUSED ", style="bold black on yellow")
        right.append("  ")
    if state.kiosk:
        right.append(" AUTO ", style="bold black on bright_cyan")
        right.append("  ")
    if state.quantile:
        right.append(" q90 ", style="bold black on bright_magenta")
        right.append("  ")
    right.append(f"{state.hz} Hz", style="grey70")
    right.append(f"  frame {state.stats.frames}", style="grey70")
    if state.total_frames:
        right.append(f"/{state.total_frames}", style="grey50")

    bar.add_row(left, mid, right)
    return Panel(bar, box=box.HEAVY, border_style="bright_blue", padding=(0, 1))


def link_panel(state: "Demo", width: int) -> Panel:
    pred    = state.prediction
    spacing = state.client.profile["mec_spacing_m"]
    mec_up  = bool(pred["mec_available"])

    body = Table.grid(expand=True)
    body.add_column()
    body.add_row(road_view(pred["dist_m"], spacing, max(24, width - 40), mec_up))

    facts = Text()
    facts.append("distance ", style="grey58")
    facts.append(f"{pred['dist_m']:>7,.0f} m", style="bright_white")
    facts.append("   RSSI ", style="grey58")
    facts.append(f"{pred['rssi_dbm']:>7.1f} dBm ", style="bright_white")
    facts.append_text(signal_bars(pred["rssi_dbm"]))
    facts.append("   MEC ", style="grey58")
    facts.append("reachable" if mec_up else "UNREACHABLE",
                 style="bright_green" if mec_up else "bold red")
    facts.append("   cell spacing ", style="grey58")
    facts.append(f"{spacing / 1000:g} km", style="bright_white")
    if state.fault_label:
        facts.append(f"   {state.fault_label} ", style="bold white on red")
        facts.append(f" {state.fault_left} frames left", style="red")
    body.add_row(facts)

    return Panel(body, title="[grey70]radio link[/]", title_align="left",
                 box=box.ROUNDED, border_style="grey35", padding=(0, 1))


def decision_panel(state: "Demo", big: bool) -> Panel:
    mode  = state.mode
    row   = state.table[mode]
    style = MODE_STYLE[mode]

    detail = Table.grid(padding=(0, 1))
    detail.add_column()

    headline = Text()
    headline.append(f"MODE {mode}", style=f"bold {style}")
    headline.append(f"   {MODE_TIER[mode]}", style="bright_white")
    headline.append(f"   {row['model']}", style="grey70")
    detail.add_row(headline)

    numbers = Text()
    total = row.get("total_ms")
    numbers.append("latency ", style="grey58")
    numbers.append(f"{total:.1f} ms" if total is not None else "n/a",
                   style="bold bright_white")
    numbers.append(f" / {LATENCY_BUDGET_MS:.0f} ms budget", style="grey50")
    numbers.append("      mAP ", style="grey58")
    numbers.append(f"{row['map50_95']:.1f}", style="bold bright_white")
    numbers.append("      accuracy loss ", style="grey58")
    numbers.append(f"{row['acc_loss'] * 100:.1f}%",
                   style="bold red" if row["acc_loss"] > ACC_LOSS_BUDGET else "bold bright_white")
    numbers.append(f" / {ACC_LOSS_BUDGET * 100:.0f}% budget", style="grey50")
    detail.add_row(numbers)

    why = Text()
    why.append("why  ", style="grey58")
    why.append(REASON_TEXT.get(state.reason, state.reason),
               style="bold red" if state.reason == "fallback_no_feasible_mode" else "bright_white")
    detail.add_row(why)
    detail.add_row(Text(f"flag  {state.reason}", style="grey42"))

    if big:
        layout = Table.grid(padding=(0, 3))
        layout.add_column(vertical="middle")
        layout.add_column(vertical="middle")
        layout.add_row(big_mode(mode), detail)
        body = layout
    else:
        body = detail

    degraded = state.reason == "fallback_no_feasible_mode"
    return Panel(
        body,
        title="[bold]SELECTED EXECUTION TIER[/]" if not degraded
              else "[bold red]DEGRADED — ON-BOARD FALLBACK[/]",
        title_align="left",
        box=box.DOUBLE,
        border_style="red" if degraded else style,
        padding=(0, 1),
    )


def modes_table(state: "Demo", wide: bool) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, padding=(0, 1))
    t.add_column("", width=1)
    t.add_column("Tier", style="grey70", no_wrap=True)
    t.add_column("Hardware / model", no_wrap=True)
    t.add_column("Inf", justify="right", width=6)
    t.add_column("Net", justify="right", width=7)
    t.add_column("Total", justify="right", width=8)
    if wide:
        t.add_column(f"latency vs {LATENCY_BUDGET_MS:.0f} ms", width=22)
    t.add_column("mAP", justify="right", width=5)
    t.add_column("Loss", justify="right", width=6)
    t.add_column("Cap", justify="right", width=5)
    t.add_column("Status", no_wrap=True, min_width=22)

    for mode, row in state.table.items():
        feas     = state.feasibility[mode]
        selected = mode == state.mode
        net      = row.get("network_ms")
        total    = row.get("total_ms")
        style    = MODE_STYLE[mode]

        if selected:
            status = Text("◀ SELECTED", style=f"bold {style}")
        elif feas["feasible"]:
            status = Text("feasible", style="grey58")
        else:
            status = Text(f"✗ {pretty_violation(feas['reason'])}", style="grey46")

        cells = [
            Text("▌", style=f"bold {style}" if selected else "grey30"),
            Text(f"{mode}  {MODE_TIER[mode]}", style=style if selected else "grey70"),
            Text(f"{row['hw']} / {row['model']}",
                 style="bright_white" if selected else "grey62"),
            Text(f"{row['inference_ms']:.1f}", style="grey70"),
            Text(f"{net:.1f}" if net is not None else "—",
                 style="grey70" if net is not None else "red"),
            Text(f"{total:.1f}" if total is not None else "—",
                 style="bold bright_white" if selected else "grey70"),
        ]
        if wide:
            cells.append(budget_bar(total, LATENCY_BUDGET_MS))
        cells += [
            Text(f"{row['map50_95']:.1f}", style="grey70"),
            Text(f"{row['acc_loss'] * 100:.1f}%",
                 style="red" if row["acc_loss"] > ACC_LOSS_BUDGET else "grey70"),
            Text(f"{row['capacity_pct']:.0f}%", style="grey70"),
            status,
        ]
        t.add_row(*cells)

    return Panel(t, title="[grey70]execution tiers[/]", title_align="left",
                 box=box.ROUNDED, border_style="grey35", padding=(0, 1))


def budgets_panel(state: "Demo") -> Panel:
    row = state.table[state.mode]
    t   = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style="grey58", width=9)
    t.add_column(width=22)
    t.add_column(justify="right", width=16)

    total = row.get("total_ms")
    t.add_row("latency", budget_bar(total, LATENCY_BUDGET_MS),
              Text(f"{total:.1f} / {LATENCY_BUDGET_MS:.0f} ms" if total is not None else "n/a",
                   style="bright_white"))
    t.add_row("acc. loss", budget_bar(row["acc_loss"] * 100, ACC_LOSS_BUDGET * 100),
              Text(f"{row['acc_loss'] * 100:.1f} / {ACC_LOSS_BUDGET * 100:.0f} %",
                   style="bright_white"))
    t.add_row("capacity", budget_bar(row["capacity_pct"], CAPACITY_LIMIT_PCT),
              Text(f"{row['capacity_pct']:.0f} / {CAPACITY_LIMIT_PCT:.0f} %",
                   style="bright_white"))
    return Panel(t, title="[grey70]QoS budgets (selected tier)[/]", title_align="left",
                 box=box.ROUNDED, border_style="grey35", padding=(0, 1))


def stats_panel(state: "Demo") -> Panel:
    s = state.stats
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="grey58")
    t.add_column(justify="left")
    t.add_column(justify="right", style="grey58")
    t.add_column(justify="left")

    t.add_row("uptime", Text(s.uptime, style="bright_white"),
              "frames", Text(f"{s.frames:,}", style="bright_white"))
    t.add_row("mean latency", Text(f"{s.mean_latency:.1f} ms", style="bright_white"),
              "switches", Text(str(s.switches), style="bright_white"))
    t.add_row("deadline misses", Text(str(s.deadline_miss),
                                      style="red" if s.deadline_miss else "bright_white"),
              "MEC reachable",
              Text(f"{100.0 * s.mec_up / s.frames if s.frames else 0:.0f}%", style="bright_white"))
    t.add_row("QoS compliance", Text(f"{s.compliance:.1f}%",
                                     style="bright_green" if s.compliance > 95 else "yellow"),
              "", Text(""))

    dwell = Text()
    dwell.append("tier dwell  ", style="grey58")
    for mode in "ABCD":
        pct = 100.0 * s.dwell[mode] / s.frames if s.frames else 0.0
        dwell.append(f"{mode} {pct:.0f}%  ", style=MODE_STYLE[mode] if pct else "grey35")

    body = Table.grid()
    body.add_column()
    body.add_row(t)
    body.add_row(dwell)
    return Panel(body, title="[grey70]session[/]", title_align="left",
                 box=box.ROUNDED, border_style="grey35", padding=(0, 1))


def timeline_panel(state: "Demo", width: int) -> Panel:
    span = max(20, width - 6)
    hist = list(state.history)[-span:]

    strip = Text()
    for mode, faulted in hist:
        strip.append("█", style=f"{MODE_STYLE[mode]}{' on grey23' if faulted else ''}")
    if not hist:
        strip.append("·" * span, style="grey30")

    legend = Text()
    for mode in "ABCD":
        legend.append("█ ", style=MODE_STYLE[mode])
        legend.append(f"{mode} {MODE_TIER[mode]}   ", style="grey58")
    legend.append("shaded background = injected fault", style="grey46")

    body = Table.grid()
    body.add_column()
    body.add_row(strip)
    body.add_row(legend)
    return Panel(body, title="[grey70]decision timeline (oldest → newest)[/]",
                 title_align="left", box=box.ROUNDED, border_style="grey35",
                 padding=(0, 1))


def footer(state: "Demo") -> Text:
    tag = TAGLINES[int(time.monotonic() / 9) % len(TAGLINES)]
    out = Text()
    out.append(f" {tag}\n", style="italic grey58")
    out.append(" 1-5", style="bold bright_white"); out.append(" scenario   ", style="grey58")
    out.append("o", style="bold bright_white");    out.append(" MEC outage   ", style="grey58")
    out.append("x", style="bold bright_white");    out.append(" blackout   ", style="grey58")
    out.append("c", style="bold bright_white");    out.append(" cloud spike   ", style="grey58")
    out.append("space", style="bold bright_white"); out.append(" pause   ", style="grey58")
    out.append("+/-", style="bold bright_white");  out.append(" speed   ", style="grey58")
    out.append("k", style="bold bright_white");    out.append(" auto-pilot   ", style="grey58")
    out.append("r", style="bold bright_white");    out.append(" reset   ", style="grey58")
    out.append("q", style="bold bright_white");    out.append(" quit", style="grey58")
    return out


# ── the demo itself ──────────────────────────────────────────────────────────

class Demo:
    """Owns the closed loop, the session counters, and the dashboard state."""

    KIOSK_SCENARIO_S = 26.0
    KIOSK_EVENT_S    = 17.0
    KIOSK_EVENTS     = ["blackout", "mec_outage", "cloud_spike"]

    def __init__(
        self,
        scenario:     str   = "suburban",
        hz:           int   = 5,
        frames:       int   = 0,
        outage_at:    int   = 40,
        outage_dur:   int   = 15,
        dwell:        int   = 1,
        quantile:     bool  = False,
        kiosk:        bool  = False,
        ema_alpha:    float = 0.3,
    ) -> None:
        self.scenario     = scenario
        self.hz           = max(1, hz)
        self.total_frames = frames                 # 0 == run until quit
        self.outage_at    = outage_at
        self.outage_dur   = outage_dur
        self.quantile     = quantile
        self.kiosk        = kiosk
        self.ema_alpha    = ema_alpha

        self.client = InferenceClient(
            scenario=scenario,
            hz=self.hz,
            quantile_path=QUANTILE_PATH if quantile else None,
        )
        self.pt         = PerformanceTable()
        self.stabilizer = DwellTimeStabilizer(k=max(1, dwell))

        self.paused     = False
        self.running    = True
        self.injection: Injection | None = None
        self.stats      = Stats()
        self.history: deque[tuple[str, bool]] = deque(maxlen=600)

        self._ema_mec: float | None = None
        self._ema_cld: float | None = None
        self._prev_mode: str | None = None
        self._next_scenario_at = time.monotonic() + self.KIOSK_SCENARIO_S
        self._next_event_at    = time.monotonic() + self.KIOSK_EVENT_S
        self._event_cycle      = 0

        self.step()                                # so frame 1 has a full table

    # ── simulation ──────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance one frame: predict, smooth, score the tiers, decide."""
        pred = self.client.tick()

        frame_no = self.stats.frames + 1
        if (self.outage_at > 0
                and self.injection is None
                and frame_no == self.outage_at):
            self.injection = Injection("blackout", self.outage_dur)

        self.fault_label:  str | None = None
        self.fault_left:   int         = 0
        faulted = self.injection is not None
        if self.injection:
            pred = self.injection.apply(pred)
            self.injection.frames_left -= 1
            self.fault_label = self.injection.label
            self.fault_left  = self.injection.frames_left
            if self.injection.frames_left <= 0:
                self.injection = None

        # Risk-aware gating: score the tiers on the q90 upper bound instead of
        # the conditional mean when --quantile is on.
        mec_raw = pred.get("mec_network_ms_q") if self.quantile else pred["mec_network_ms"]
        cld_raw = pred.get("cloud_network_ms_q") if self.quantile else pred["cloud_network_ms"]
        mec_raw = mec_raw if mec_raw is not None else pred["mec_network_ms"]
        cld_raw = cld_raw if cld_raw is not None else pred["cloud_network_ms"]

        a         = self.ema_alpha
        net, cap  = {}, {}

        if pred["mec_available"]:
            self._ema_mec = mec_raw if self._ema_mec is None else a * mec_raw + (1 - a) * self._ema_mec
            net["C"], cap["C"] = self._ema_mec, pred["mec_capacity_pct"]
        else:
            self._ema_mec = None
            self.pt.mark_mec_unavailable()

        self._ema_cld = cld_raw if self._ema_cld is None else a * cld_raw + (1 - a) * self._ema_cld
        net["D"], cap["D"] = self._ema_cld, pred["cloud_capacity_pct"]
        self.pt.update(net, cap)

        table            = self.pt.get()
        proposed, reason = select_mode(table)
        # A safety fallback must engage immediately; ordinary preference
        # changes go through the dwell-time filter.
        mode = self.stabilizer.apply(proposed, force=(reason == "fallback_no_feasible_mode"))

        self.prediction  = pred
        self.table       = table
        self.feasibility = explain(table)
        self.mode        = mode
        self.reason      = reason if mode == proposed else "dwell_hold_previous_tier"

        s = self.stats
        s.frames += 1
        s.dwell[mode] += 1
        s.mec_up += int(bool(pred["mec_available"]))
        if self._prev_mode is not None and mode != self._prev_mode:
            s.switches += 1
        self._prev_mode = mode

        row   = table[mode]
        total = row.get("total_ms")
        if total is not None:
            s.latency_sum += total
            if total > LATENCY_BUDGET_MS:
                s.deadline_miss += 1
        if row["acc_loss"] > ACC_LOSS_BUDGET:
            s.acc_breach += 1

        self.history.append((mode, faulted))

    # ── presenter actions ───────────────────────────────────────────────────

    def set_scenario(self, scenario: str) -> None:
        self.scenario = scenario
        self.client.set_scenario(scenario)
        self._ema_mec = self._ema_cld = None
        self.stabilizer.reset()

    def set_hz(self, hz: int) -> None:
        self.hz = max(1, min(30, hz))
        self.client.hz = self.hz

    def inject(self, kind: str, frames: int | None = None) -> None:
        self.injection = Injection(kind, frames or max(8, self.hz * 3))

    def handle_key(self, key: str) -> None:
        if key in "12345":
            self.set_scenario(SCENARIO_ORDER[int(key) - 1])
        elif key == " ":
            self.paused = not self.paused
        elif key in ("+", "="):
            self.set_hz(self.hz + 1)
        elif key in ("-", "_"):
            self.set_hz(self.hz - 1)
        elif key == "o":
            self.inject("mec_outage")
        elif key == "x":
            self.inject("blackout")
        elif key == "c":
            self.inject("cloud_spike")
        elif key == "k":
            self.kiosk = not self.kiosk
            self._next_scenario_at = time.monotonic() + self.KIOSK_SCENARIO_S
            self._next_event_at    = time.monotonic() + self.KIOSK_EVENT_S
        elif key == "r":
            self.stats.reset()
            self.history.clear()
        elif key in ("q", "Q", "\x03", "\x04"):
            self.running = False

    def kiosk_tick(self) -> None:
        """Unattended booth behaviour: rotate scenarios, stage faults."""
        if not self.kiosk or self.paused:
            return
        now = time.monotonic()
        if now >= self._next_scenario_at:
            nxt = SCENARIO_ORDER[(SCENARIO_ORDER.index(self.scenario) + 1) % len(SCENARIO_ORDER)]
            self.set_scenario(nxt)
            self._next_scenario_at = now + self.KIOSK_SCENARIO_S
        if now >= self._next_event_at and self.injection is None:
            self.inject(self.KIOSK_EVENTS[self._event_cycle % len(self.KIOSK_EVENTS)])
            self._event_cycle += 1
            self._next_event_at = now + self.KIOSK_EVENT_S

    # ── rendering ───────────────────────────────────────────────────────────

    def render(self, console: Console) -> Group:
        width, height = console.size
        wide  = width >= 118
        parts = [header_panel(self), link_panel(self, width)]
        parts.append(decision_panel(self, big=height >= 30 and width >= 88))
        parts.append(modes_table(self, wide))

        if height >= 32:
            side = Table.grid(expand=True, padding=(0, 1))
            side.add_column(ratio=1)
            side.add_column(ratio=1)
            side.add_row(budgets_panel(self), stats_panel(self))
            parts.append(side)
        if height >= 38:
            parts.append(timeline_panel(self, width))
        parts.append(footer(self))
        return Group(*parts)


# ── artefact bootstrap ───────────────────────────────────────────────────────

def ensure_artifacts(console: Console, skip_train: bool) -> None:
    """Generate the dataset and train the predictor if they are not on disk."""
    if skip_train:
        missing = [str(p) for p in (DATA_PATH, MODEL_PATH) if not p.exists()]
        if missing:
            console.print(f"[red]--no-train given but missing:[/] {', '.join(missing)}")
            sys.exit(1)
        return

    if DATA_PATH.exists():
        console.print(f"  dataset  [green]ok[/]   {DATA_PATH}  "
                      f"({DATA_PATH.stat().st_size // 1024:,} KB)")
    else:
        console.print("  dataset  [yellow]generating 50k physics-based samples …[/]")
        generate_dataset()

    if MODEL_PATH.exists():
        console.print(f"  model    [green]ok[/]   {MODEL_PATH}  "
                      f"({MODEL_PATH.stat().st_size // 1024 // 1024:,} MB)")
    else:
        console.print("  model    [yellow]training predictor (this takes ~1 min) …[/]")
        train()


def summary(console: Console, demo: Demo) -> None:
    s = demo.stats
    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="grey58", justify="right")
    t.add_column(style="bright_white")
    t.add_row("frames", f"{s.frames:,}")
    t.add_row("runtime", s.uptime)
    t.add_row("mode switches", str(s.switches))
    t.add_row("mean latency", f"{s.mean_latency:.1f} ms")
    t.add_row("deadline misses", str(s.deadline_miss))
    t.add_row("accuracy-budget breaches", f"{s.acc_breach} (on-board fallback frames)")
    t.add_row("QoS compliance", f"{s.compliance:.1f}%")
    t.add_row("tier dwell", "  ".join(
        f"{m} {100.0 * s.dwell[m] / s.frames if s.frames else 0:.0f}%" for m in "ABCD"))
    console.print(Panel(t, title="[bold]session summary[/]", box=box.ROUNDED,
                        border_style="bright_blue", expand=False))


def run(**kwargs) -> None:
    """Boot the artefacts, then drive the dashboard until the run ends."""
    skip_train = kwargs.pop("skip_train", False)
    no_input   = kwargs.pop("no_input", False)
    console    = Console()

    console.print()
    console.print("  [bold bright_white]HUA · QoS-aware AI inference offloading[/]"
                  "  ·  IEEE CloudNet 2026 demo")
    ensure_artifacts(console, skip_train)
    console.print("  [grey58]loading predictor …[/]")

    demo = Demo(**kwargs)
    console.clear()

    with KeyReader(enabled=not no_input) as keys, \
         Live(demo.render(console), console=console, screen=True,
              refresh_per_second=20, transient=False) as live:
        while demo.running:
            if demo.total_frames and demo.stats.frames >= demo.total_frames:
                break
            deadline = time.monotonic() + 1.0 / demo.hz
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                key = keys.get(min(0.05, remaining))
                if key:
                    demo.handle_key(key)
                    if not demo.running:
                        break
                live.update(demo.render(console))
            if not demo.running:
                break
            demo.kiosk_tick()
            if not demo.paused:
                demo.step()
            live.update(demo.render(console))

    console.print()
    summary(console, demo)
    console.print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="IEEE CloudNet 2026 — live QoS-aware offloading demonstrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--scenario", default="suburban", choices=SCENARIO_ORDER,
                    help="Road environment to start in")
    ap.add_argument("--hz", type=int, default=5, help="Frames per second")
    ap.add_argument("--frames", type=int, default=0,
                    help="Frames to simulate (0 = run until 'q')")
    ap.add_argument("--outage-at", type=int, default=40,
                    help="Frame that triggers the scripted blackout (0 = none)")
    ap.add_argument("--outage-dur", type=int, default=15,
                    help="Length of the scripted blackout, in frames")
    ap.add_argument("--dwell", type=int, default=1,
                    help="Min-dwell hysteresis, in frames (1 = off)")
    ap.add_argument("--quantile", action="store_true",
                    help="Gate on the q90 RTT upper bound instead of the mean")
    ap.add_argument("--kiosk", action="store_true",
                    help="Unattended booth loop: run forever, rotate scenarios, stage faults")
    ap.add_argument("--no-train", dest="skip_train", action="store_true",
                    help="Refuse to build missing artefacts instead of generating them")
    ap.add_argument("--no-input", action="store_true",
                    help="Ignore the keyboard (for logging or a non-interactive terminal)")
    args = ap.parse_args()

    if args.kiosk:
        args.frames    = 0
        args.outage_at = 0

    if args.quantile and not QUANTILE_PATH.exists():
        print(f"--quantile requested but {QUANTILE_PATH} is missing; "
              f"train it with 'python -m src.predictor.quantile'.", file=sys.stderr)
        sys.exit(1)

    try:
        run(
            scenario=args.scenario, hz=args.hz, frames=args.frames,
            outage_at=args.outage_at, outage_dur=args.outage_dur,
            dwell=args.dwell, quantile=args.quantile, kiosk=args.kiosk,
            skip_train=args.skip_train, no_input=args.no_input,
        )
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
