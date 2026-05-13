"""
HUA QoS-Aware AI Offloading — end-to-end demo — IEEE CloudNet 2026.

Runs the full closed-loop pipeline and displays a single live table that
updates in-place on every frame:

    1. Generate the dataset (if not already present).
    2. Train the ML predictor (if model not already saved).
    3. Stream per-frame latency predictions for a simulated vehicle.
    4. Update the live performance table with each prediction.
    5. Call the QoS-aware optimizer and overwrite the table in-place.

The demo injects a simulated connectivity outage (tunnel / dense urban canyon)
to demonstrate the on-board GPU fallback tier (Modes A/B).  Under normal
conditions Modes A and B are excluded by the 15 % accuracy-loss budget
(YOLOv8n drops ~21 % vs the YOLOv8l cloud baseline), so offloading is always
preferred when the network is available.  The outage forces both MEC and cloud
to exceed their latency/availability thresholds, activating Mode B as the
emergency on-board fallback.

Usage::

    python demo.py
    python demo.py --scenario highway --hz 5 --frames 200
    python demo.py --scenario urban_dense --hz 10 --frames 500 --no-train
    python demo.py --outage-at 0                 # disable the injected outage
    python demo.py --outage-at 30 --outage-dur 20
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DATA_PATH, MODEL_PATH, SCENARIOS
from src.dataset.generate     import generate_dataset
from src.predictor.model      import train
from src.predictor.client     import InferenceClient
from src.offloading.table     import PerformanceTable
from src.offloading.optimizer import explain, select_mode

_ANSI_RESET  = "\033[0m"
_ANSI_BOLD   = "\033[1m"
_ANSI_GREEN  = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED    = "\033[91m"
_ANSI_CYAN   = "\033[96m"
_ANSI_DIM    = "\033[2m"

_MODE_COLOR = {
    "A": _ANSI_YELLOW,
    "B": _ANSI_CYAN,
    "C": _ANSI_GREEN,
    "D": _ANSI_RED,
}

_SEP  = "─" * 74
_SEP2 = "━" * 74

_TABLE_LINES = 15


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_ANSI_RESET}"


def _ensure_dataset() -> None:
    if DATA_PATH.exists():
        print(f"  Dataset found  → {DATA_PATH}  ({DATA_PATH.stat().st_size // 1024:,} KB)")
        return
    print("  Dataset not found — generating …")
    generate_dataset()


def _ensure_model() -> None:
    if MODEL_PATH.exists():
        print(f"  Model found    → {MODEL_PATH}  ({MODEL_PATH.stat().st_size // 1024:,} KB)")
        return
    print("  Model not found — training …")
    train()


def _build_table(
    frame_n:      int,
    total_frames: int,
    scenario:     str,
    hz:           int,
    prediction:   dict,
    table:        dict,
    mode:         str,
    reason:       str,
    feasibility:  dict,
    switch_count: int,
    in_outage:    bool = False,
) -> list[str]:
    """
    Build the live table as a fixed list of exactly ``_TABLE_LINES`` strings.

    Returning lines instead of printing lets ``_draw`` overwrite them in-place
    on every frame using ANSI cursor-up escape codes.
    """
    rssi   = prediction["rssi_dbm"]
    dist   = prediction["dist_m"]
    mec_ok = "YES" if prediction["mec_available"] else _color("NO", _ANSI_RED)

    outage_tag = (
        f"  {_color('[OUTAGE — on-board fallback]', _ANSI_RED + _ANSI_BOLD)}"
        if in_outage else ""
    )

    lines = []

    lines.append(_color(_SEP2, _ANSI_BOLD))
    lines.append(
        _color("  HUA  ·  QoS-Aware AI Offloading  ·  IEEE CloudNet 2026", _ANSI_BOLD)
    )
    lines.append(_color(_SEP2, _ANSI_BOLD))
    lines.append(
        f"  Scenario: {_color(f'{scenario}', _ANSI_BOLD):<30}"
        f"  Frame: {frame_n:>4}/{total_frames}"
        f"  Hz: {hz}"
        f"  Switches: {_color(str(switch_count), _ANSI_BOLD)}"
        + outage_tag
    )
    lines.append(
        f"  Dist: {dist:>7.0f} m"
        f"  │  RSSI: {rssi:>7.1f} dBm"
        f"  │  MEC: {mec_ok}"
        f"  │  Budget: latency ≤ 100ms  acc.loss ≤ 15%"
    )
    lines.append(_color(_SEP, _ANSI_DIM))
    lines.append(
        f"  {'Mode':<5} {'Model':<10} {'HW':<10}"
        f" {'Inf':>5} {'Net':>7} {'Total':>7}"
        f" {'mAP':>6} {'AccLoss':>8} {'Cap%':>5}   Status"
    )
    lines.append("  " + "─" * 70)

    for m, row in table.items():
        net_s = f"{row['network_ms']:.1f}" if row.get("network_ms") is not None else "N/A"
        tot_s = f"{row['total_ms']:.1f}"   if row.get("total_ms")   is not None else "N/A"
        feas  = feasibility[m]

        if m == mode:
            status = _color("◄ SELECTED", _MODE_COLOR[m] + _ANSI_BOLD)
        elif feas["feasible"]:
            status = _color("feasible", _ANSI_DIM)
        else:
            status = _color(f"✗ {feas['reason']}", _ANSI_DIM)

        row_str = (
            f"  {m:<5} {row['model']:<10} {row['hw']:<10}"
            f" {row['inference_ms']:>5.1f} {net_s:>7} {tot_s:>7}"
            f" {row['map50_95']:>6.1f} {row['acc_loss']*100:>7.1f}%"
            f" {row['capacity_pct']:>4.0f}%   {status}"
        )
        lines.append(_color(row_str, _MODE_COLOR[m]) if m == mode else row_str)

    lines.append(_color(_SEP, _ANSI_DIM))
    lines.append(
        f"  {_color('►', _ANSI_BOLD)} "
        f"{_color(f'Mode {mode}', _MODE_COLOR[mode] + _ANSI_BOLD)}"
        f"  [{table[mode]['model']} @ {table[mode]['hw']}]"
        f"  {_color(reason, _ANSI_DIM)}"
    )
    lines.append(_color(_SEP2, _ANSI_BOLD))

    assert len(lines) == _TABLE_LINES, f"expected {_TABLE_LINES} lines, got {len(lines)}"
    return lines


def _draw(lines: list[str], first: bool) -> None:
    """Overwrite the previous table in-place, or print fresh on the first frame."""
    if not first:
        sys.stdout.write(f"\033[{_TABLE_LINES}A\r")
    for line in lines:
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.flush()


def run(
    scenario:    str  = "urban_dense",
    hz:          int  = 5,
    frames:      int  = 100,
    skip_train:  bool = False,
    outage_at:   int  = 40,
    outage_dur:  int  = 15,
) -> None:
    """
    Run the end-to-end demo loop with a single in-place updating table.

    Parameters
    ----------
    scenario:
        Road environment profile for the inference client.
    hz:
        Frames per second (controls simulation speed and sleep interval).
    frames:
        Total number of frames to simulate.
    skip_train:
        Skip dataset/model generation if artefacts already exist on disk.
    outage_at:
        Frame number at which to inject a simulated connectivity outage.
        Set to 0 to disable.  During the outage the MEC is marked
        unreachable and the cloud RTT is inflated to exceed the latency
        budget, forcing the system to fall back to the on-board GPU (Mode B).
    outage_dur:
        Duration of the injected outage in frames.
    """
    print()
    print("  Checking artefacts …")
    if not skip_train:
        _ensure_dataset()
        _ensure_model()
    else:
        if not DATA_PATH.exists() or not MODEL_PATH.exists():
            print("  ERROR: --no-train requested but dataset or model is missing.")
            sys.exit(1)
        print(f"  Dataset : {DATA_PATH}")
        print(f"  Model   : {MODEL_PATH}")
    print()

    _EMA_ALPHA   = 0.3
    _ema_mec_ms  = None
    _ema_cld_ms  = None

    client       = InferenceClient(scenario=scenario, hz=hz)
    pt           = PerformanceTable()
    prev_mode    = None
    switch_count = 0
    first_frame  = True

    _outage_end = (outage_at + outage_dur - 1) if outage_at > 0 else -1

    for frame_n in range(1, frames + 1):
        prediction   = client.tick()
        network_ms   = {}
        capacity_pct = {}

        # Inject simulated connectivity outage (tunnel / dense urban canyon).
        # MEC is marked unreachable; cloud RTT is inflated so total latency
        # exceeds the 100 ms budget (4 ms inference + >96 ms network), forcing
        # the optimizer to fall back to the on-board GPU (Mode B).
        in_outage = outage_at > 0 and outage_at <= frame_n <= _outage_end
        if in_outage:
            prediction = dict(prediction)
            prediction["mec_available"]   = False
            prediction["cloud_network_ms"] = 200.0  # 4 + 200 >> 100 ms budget

        raw_mec = prediction["mec_network_ms"]
        raw_cld = prediction["cloud_network_ms"]

        if prediction["mec_available"] and raw_mec is not None:
            _ema_mec_ms = (raw_mec if _ema_mec_ms is None
                           else _EMA_ALPHA * raw_mec + (1 - _EMA_ALPHA) * _ema_mec_ms)
            network_ms["C"]   = _ema_mec_ms
            capacity_pct["C"] = prediction["mec_capacity_pct"]
        else:
            _ema_mec_ms = None
            pt.mark_mec_unavailable()

        _ema_cld_ms = (raw_cld if _ema_cld_ms is None
                       else _EMA_ALPHA * raw_cld + (1 - _EMA_ALPHA) * _ema_cld_ms)
        network_ms["D"]   = _ema_cld_ms
        capacity_pct["D"] = prediction["cloud_capacity_pct"]

        if network_ms or capacity_pct:
            pt.update(network_ms, capacity_pct)

        table        = pt.get()
        mode, reason = select_mode(table)
        feasibility  = explain(table)

        if prev_mode is not None and mode != prev_mode:
            switch_count += 1

        lines = _build_table(
            frame_n=frame_n,
            total_frames=frames,
            scenario=scenario,
            hz=hz,
            prediction=prediction,
            table=table,
            mode=mode,
            reason=reason,
            feasibility=feasibility,
            switch_count=switch_count,
            in_outage=in_outage,
        )
        _draw(lines, first=first_frame)
        first_frame = False
        prev_mode   = mode

        time.sleep(1.0 / hz)

    print(f"\n  Done — {frames} frames simulated, {switch_count} mode switches.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CloudNet 2026 end-to-end demo.")
    ap.add_argument("--scenario",    default="urban_dense", choices=list(SCENARIOS))
    ap.add_argument("--hz",          type=int, default=5,   help="Frames per second")
    ap.add_argument("--frames",      type=int, default=100, help="Number of frames to simulate")
    ap.add_argument("--no-train",    dest="skip_train", action="store_true",
                    help="Skip dataset/model generation (requires existing artefacts)")
    ap.add_argument("--outage-at",   type=int, default=40,
                    help="Frame at which to inject a connectivity outage (0 = disabled)")
    ap.add_argument("--outage-dur",  type=int, default=15,
                    help="Duration of the injected outage in frames")
    args = ap.parse_args()

    run(
        scenario   = args.scenario,
        hz         = args.hz,
        frames     = args.frames,
        skip_train = args.skip_train,
        outage_at  = args.outage_at,
        outage_dur = args.outage_dur,
    )
