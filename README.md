# HUA — QoS-Aware AI Offloading for Connected Vehicles

**IEEE CloudNet 2026 · Demo & Artifact Submission**  
HUA · Technical University of Denmark (DTU)

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
git clone <repo-url> && cd cloudnet
pip install -r requirements.txt

# 1 — generate dataset
python -m src.dataset.generate

# 2 — train the ML predictor
python -m src.predictor.model

# 3 — run the end-to-end demo
python demo.py --scenario urban_dense --hz 5 --frames 100
```

If dataset and model are already present:

```bash
python demo.py --no-train --scenario highway --hz 10 --frames 200
```

---

## Execution Modes

| Mode | Hardware    | Model    | Inf. (ms) | Net. (ms) | Total (ms) | mAP@0.5:0.95 |
|------|-------------|----------|-----------|-----------|------------|--------------|
| A    | Car GPU 1   | YOLOv8n  | 8         | 0         | 8          | 37.3         |
| B    | Car GPU 2   | YOLOv8n  | 5         | 0         | 5          | 37.3         |
| C    | MEC server  | YOLOv8m  | 6         | ~18       | ~24        | 50.2         |
| D    | Cloud       | YOLOv8l  | 4         | ~60       | ~84        | 52.9         |

**QoS budgets:** latency ≤ 100 ms · accuracy loss ≤ 15% · capacity < 95%

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
  Frame    42  │  dist:    823 m  │  RSSI:  -68.4 dBm  │  MEC reachable: YES
────────────────────────────────────────────────────────────────────────
  Mode  Model      HW          Inf     Net    Total    mAP  AccLoss   Cap%   Status
  ─────────────────────────────────────────────────────────────────────────
  A     yolov8n    Car GPU 1    8.0     0.0     8.0   37.3    14.7%     0%   feasible
  B     yolov8n    Car GPU 2    5.0     0.0     5.0   37.3    14.7%     0%   feasible
  C     yolov8m    MEC          6.0    22.1    28.1   50.2     5.1%    43%   feasible
  D     yolov8l    Cloud        4.0    63.4    67.4   52.9     0.0%    28%   ◄ SELECTED

  ► Decision : Mode D  [yolov8l @ Cloud]  max_accuracy_feasible
```

---

## License

MIT

---


