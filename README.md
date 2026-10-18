# QoS-Aware AI Inference Offloading for Vehicular Edge Computing

**IEEE CloudNet 2026 · Demo & Artifact Submission**  
Dept. of Informatics and Telematics, Harokopio University of Athens (HUA)

**Conference presentation:** [7-slide PDF](presentation/CLOUDNET2026_Demo_Gyftodimos.pdf)
· [Greek presenter notes](presentation/PRESENTER_NOTES_EL.md)
· [Demo walkthrough and preparation](presentation/README.md)

---

## Overview

This is an open simulation framework for studying **where to run an AI
inference task** (YOLOv8 object detection) in a connected vehicle scenario.
A vehicle can execute on its own GPU, offload to a nearby MEC server, or send
frames to the cloud — each tier offering a different latency/accuracy trade-off.

The framework has three components:

1. **Physics-based dataset generator** — 50,000 labeled samples across five
   road environments, derived from a 3GPP path-loss + Shannon throughput model.
2. **Multi-output ML predictor** — predicts MEC and cloud network latency and
   capacity utilisation in real time from observable vehicle and radio state.
3. **QoS-aware offloading optimizer** — closes the loop by selecting the best
   execution mode per frame within hard latency and accuracy-loss budgets.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      demo.py  (entry point)                │
└──────────┬──────────────────────────────────┬──────────────┘
           │                                  │
  ┌────────▼────────┐                ┌────────▼────────┐
  │  src/dataset/   │                │  src/detection/ │
  │  generate.py    │                │  benchmark.py   │
  │                 │                │  detect_video.py│
  │  3GPP path-loss │                │  YOLOv8 n/m/l   │
  │  Shannon BW     │                └─────────────────┘
  │  5 scenarios    │
  └────────┬────────┘
           │  network_dataset.csv
  ┌────────▼────────┐
  │  src/predictor/ │
  │  model.py       │  RandomForest (regression)
  │  client.py      │  GradientBoosting (classification)
  └────────┬────────┘
           │  per-frame predictions
  ┌────────▼────────┐
  │  src/offloading/│
  │  table.py       │  live performance table
  │  optimizer.py   │  select_mode() → Mode A/B/C/D
  └─────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/icsa-hua/CLOUDNET2026-QoS-Offloading.git
cd CLOUDNET2026-QoS-Offloading
pip install -r requirements.txt

# 1 — generate dataset
python -m src.dataset.generate

# 2 — train the ML predictor
python -m src.predictor.model

# 3 — run the end-to-end demo
python demo.py
```

Steps 1 and 2 are optional: `demo.py` generates the dataset and trains the
predictor on first run if they are missing.

By default, `python demo.py` keeps running until you press `q` (no Enter needed).
The scripted blackout still starts at frame 40; use `--outage-at 0` for manual faults only.

```bash
python demo.py --kiosk                       # unattended booth loop
python demo.py --scenario highway --hz 10    # pick a road environment
python demo.py --frames 0                    # run until 'q'
python demo.py --frames 100                  # optional finite run
python demo.py --quantile                    # gate on the q90 RTT upper bound
python demo.py --outage-at 0                 # no scripted blackout
python demo.py --no-train                    # fail instead of building artefacts
```

### Live controls

The demo is a full-screen dashboard; every key acts immediately, no Enter.

| Key     | Action                                    |
|---------|-------------------------------------------|
| `1`–`5` | switch road scenario                      |
| `o`     | inject MEC outage — forces C → D          |
| `x`     | inject full blackout — forces C → B       |
| `c`     | inject cloud congestion                   |
| `space` | pause / resume                            |
| `+` `-` | faster / slower                           |
| `k`     | toggle kiosk auto-pilot                   |
| `r`     | reset session counters                    |
| `q`     | quit                                      |

`--kiosk` runs forever, rotates through all five scenarios, and stages the
three faults on a timer — the mode to leave running on a booth monitor.

---

## Execution Modes

| Mode | Hardware    | Model    | Inf. (ms) | Net. (ms) | Total (ms) | mAP@0.5:0.95 | Acc. Loss |
|------|-------------|----------|-----------|-----------|------------|--------------|-----------|
| A    | Car NPU     | YOLOv8n  | 8         | 0         | 8          | 37.3         | 20.8%\*   |
| B    | Car GPU     | YOLOv8n  | 5         | 0         | 5          | 37.3         | 20.8%\*   |
| C    | MEC server  | YOLOv8m  | 6         | ~18       | ~24        | 50.2         |  4.9%     |
| D    | Cloud       | YOLOv8l  | 4         | ~60       | ~84        | 52.9         |  0.0%     |

**QoS budgets:** latency ≤ 100 ms · accuracy loss ≤ 15% · capacity < 95%  
\* Modes A and B exceed the accuracy-loss budget and are only activated as emergency fallback.

---

## Dataset Schema

| Column                | Type    | Description                              |
|-----------------------|---------|------------------------------------------|
| `scenario`            | string  | Road environment type                    |
| `car_speed_kmh`       | float   | Vehicle speed (km/h)                     |
| `time_of_day_h`       | float   | Hour of day [0, 24]                      |
| `mec_distance_m`      | float   | Distance to serving MEC tower (m)        |
| `mec_rssi_dbm`        | float   | Received signal strength (dBm)           |
| `mec_channel_bw_mhz`  | float   | Channel bandwidth (MHz)                  |
| `mec_load_pct`        | float   | MEC server load (%)                      |
| `num_connected_ues`   | int     | UEs sharing the cell                     |
| `handover_active`     | binary  | Handover event flag                      |
| `payload_kb`          | float   | Frame payload size (kB)                  |
| `cloud_congestion`    | float   | WAN congestion index [0, 1]              |
| `cloud_rtt_base_ms`   | float   | Base cloud RTT (ms)                      |
| `cloud_load_pct`      | float   | Cloud server load (%)                    |
| `mec_available`       | binary  | **Target** — MEC reachability label      |
| `mec_network_ms`      | float   | **Target** — MEC RTT (ms)               |
| `mec_capacity_pct`    | float   | **Target** — MEC capacity utilisation   |
| `cloud_network_ms`    | float   | **Target** — Cloud RTT (ms)             |
| `cloud_capacity_pct`  | float   | **Target** — Cloud capacity utilisation |

---

## Road Scenarios

| Scenario       | MEC spacing | Speed range   | Frequency | BW    |
|----------------|-------------|---------------|-----------|-------|
| urban_dense    | 300 m       | 20–50 km/h    | 3.5 GHz   | 100 MHz |
| urban_sparse   | 800 m       | 30–70 km/h    | 3.5 GHz   | 100 MHz |
| suburban       | 2 000 m     | 50–90 km/h    | 2.1 GHz   | 40 MHz  |
| highway        | 5 000 m     | 80–130 km/h   | 2.1 GHz   | 20 MHz  |
| rural          | 10 000 m    | 60–100 km/h   | 0.9 GHz   | 10 MHz  |

---

## Module Reference

### `src/dataset/generate.py`
Generates the synthetic dataset. Key function: `generate_dataset(n, out, seed)`.

### `src/predictor/model.py`
Trains, evaluates, and runs the ML bundle.  
Key functions: `train()`, `evaluate()`, `predict(bundle, state)`, `load_bundle()`.

### `src/predictor/client.py`
Stateful per-frame simulator. Key class: `InferenceClient(scenario, hz)` → `.tick()`.

### `src/offloading/table.py`
Merges static hardware specs with live network predictions.  
Key class: `PerformanceTable` → `.update(network_ms, capacity_pct)` → `.get()`.

### `src/offloading/optimizer.py`
The decision function.  
`select_mode(table) → (mode, reason)` — picks the feasible mode with highest mAP.  
`explain(table) → dict` — per-mode feasibility annotations for display.

### `src/detection/benchmark.py`
YOLOv8 inference latency benchmark. Key function: `run_benchmark(n_runs, ...)`.

### `src/detection/detect_video.py`
Annotated video generation. Key function: `run_detection(source, modes, ...)`.

---

## Demo Output (example)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Frame   100  │  dist:    611 m  │  RSSI:  -55.5 dBm  │  MEC reachable: YES
────────────────────────────────────────────────────────────────────────
  Mode  Model      HW          Inf     Net    Total    mAP  AccLoss   Cap%   Status
  ─────────────────────────────────────────────────────────────────────────
  A     yolov8n    Car NPU      8.0     0.0     8.0   37.3    20.8%     0%   ✗ loss > 15%
  B     yolov8n    Car GPU      5.0     0.0     5.0   37.3    20.8%     0%   ✗ loss > 15%
  C     yolov8m    MEC          6.0    50.9    56.9   50.2     4.9%    43%   ◄ SELECTED
  D     yolov8l    Cloud        4.0    67.6    71.6   52.9     0.0%    28%   feasible, not preferred

  ► Decision : Mode C  [yolov8m @ MEC]  edge_first_preferred_over_cloud
```

---

## License

MIT

---

## Citation

```bibtex
@inproceedings{gyftodimos2026cloudnet,
  title     = {{QoS}-Aware {AI} Inference Offloading for Vehicular Edge Computing:
               A Physics-Based Dataset and Simulation Framework},
  author    = {Gyftodimos, Konstantinos and Liotou, Eirini and
               Chiotis, Kyriakos and Dimitrakopoulos, George},
  booktitle = {Proc. IEEE Int. Conf. on Cloud Networking (CloudNet)},
  year      = {2026},
  note      = {Demo \& Artifact Track}
}
```
