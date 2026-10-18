# IEEE CloudNet 2026 demo presentation

Open **CLOUDNET2026_Demo_Gyftodimos.pdf**. It contains a new illustrated perception-flow slide, the five existing presentation slides, and two annotated dashboard reference slides (eight pages total). There are no title-only or closing slides. Presenter: Konstantinos Gyftodimos, PhD candidate, Harokopio University of Athens (HUA).

## Before presenting

Use an environment with the repository dependencies installed. On the preparation machine, `/Users/konstantinosgyftodimos/miniconda3/bin/python` has the required demo packages.

```bash
python demo.py --outage-at 0
```

This disables the automatic blackout so you control the timing. Plain `python demo.py` also runs until `q`, but retains the original scripted blackout at frame 40 for 15 frames. `--frames 100` still enables a finite run. Keyboard control requires a focused interactive terminal; no Enter is needed. Increase the terminal size to at least 156 columns and 38 rows (or reduce the font) to show every dashboard panel. Avoid `--no-input` when you need `q`.

Suggested sequence: wait for C; press `o` for an MEC outage and describe D if feasible; wait for recovery; press `x` for a blackout and describe B; show the red accuracy loss and reduced compliance; wait for recovery. Each manual fault lasts approximately three seconds at an unchanged update rate. Cloud delay is smoothed, so blackout transitions need not happen on the first frame. Press space to pause a state for explanation; press it again to continue. Press `q` to finish.

The normal, cloud and blackout images are lossless captures of the actual Rich renderable produced by `Demo`, using the bundled predictor and deterministic simulator seed. They are terminal-output captures rendered directly to PNG, rather than photographs or recreated dashboard mockups. Their uptime is near zero because capture advances simulated frames without waiting in real time. Live numbers will differ.

## What the display means

- Header: selected road scenario and its radio profile; update rate (Hz); frame count. Optional indicators: PAUSED; AUTO (kiosk); q90 (upper-quantile gating).
- Radio link: diamond = vehicle position in the schematic; distance = simulated distance to MEC; RSSI = received signal strength in dBm; bars = a visual RSSI scale; reachable/unreachable = predicted MEC availability; spacing = scenario cell spacing; red label and countdown = injected fault.
- Decision: A = local NPU; B = local GPU; C = MEC; D = cloud. Model, chosen-tier latency, mAP, accuracy loss, plain-language explanation and machine-readable reason code appear beside the letter. A degraded fallback remains selected even though it fails the accuracy constraint.
- Execution table: Inf = configured inference time; Net = smoothed predicted network time; Total = their sum (all in ms); bar = total latency against the 100 ms marker; mAP = configured overall model detection score; Loss = average relative class-AP reduction across the five target classes compared with YOLOv8l; Cap = predicted utilization; Status = selected, feasible, or first failed constraint. A dash indicates an unavailable value.
- Gauges: current selected-tier latency, accuracy loss and utilization against their limits. Color indicates proximity to / violation of a limit. The capacity policy rejects values at or above 95%; the gauge itself becomes red only above its numeric budget.
- Session: wall-clock uptime; frame count; average selected-tier predicted latency; number of tier changes; predicted deadline misses; fraction of frames with MEC reachable; tier dwell = share of frames assigned to each tier. Displayed QoS compliance is calculated from latency/accuracy breach counters; it is not an independent real-network measurement and does not include capacity breaches. The implementation subtracts both counters separately, so overlapping failures would be counted twice.
- Timeline: oldest to newest; yellow A, cyan B, green C, magenta D; shaded background = injected fault. Only the recent history is visible.
- Footer: rotating explanatory tips and all controls. Keys 1–5 select dense city, outskirts, suburban, highway, rural; `o` MEC outage; `x` blackout; `c` cloud congestion; space pause; `+`/`-` rate; `k` kiosk; `r` reset counters/history; `q` quit. Reset does not restart the vehicle simulation or clear an active injection.

## Scientific scope

The paper title is **PhySDV: Physics-Grounded Data Generation and Risk-Aware Learned Offloading for QoS-Constrained Perception in Software-Defined Vehicles**. Slides use the current `new_paper/main.tex` and experiment JSON files. The 6.12 ms / 1.27 MB results concern the HistGB comparison model, while the bundled demo predictor is the existing RandomForest artifact. The q90 result compares `point_ema` with `q90_ema` in urban_sparse, without dwell filtering. The dashboard uses configured YOLOv8 model properties and predicted network conditions; it does not execute live camera inference or access real MEC/cloud infrastructure.

## Official logo sources (retrieved 18 October 2026)

- HUA logo and official variants: https://www.hua.gr/en/logos/
- HUA image: https://www.hua.gr/wp-content/uploads/2024/07/HUA-Logo-Blue-RGB.png
- IEEE CloudNet conference listing: https://www.comsoc.org/conferences-events/ieee-international-conference-cloud-networking-2026
- CloudNet image: https://www.comsoc.org/sites/default/files/styles/400wide/public/images/2025-2025-10/comsoc-conference-logos-220x80-v1-23oct25-kp-cloud-net.png.webp?itok=l_X9Xzbq
- Conference website / IEEE header image: https://cloudnet2026.ieee-cloudnet.org/ and https://cloudnet2026.ieee-cloudnet.org/sites/default/files/images/2023-2023-07/ieee%402x.png

## Rebuild

`capture_demo.py` needs the demo dependencies and Pillow. `build_presentation.py` needs ReportLab, Pillow and PyMuPDF. Both use the system Arial/Menlo fonts on macOS; adjust font paths on other platforms.

```bash
python presentation/capture_demo.py
python presentation/build_presentation.py
```

`overview.png` previews all eight pages; `slide-01.png` through `slide-08.png` are individual slide previews. Logo downloads and the generated rural illustration are included so rebuilding does not need network access. The illustration was generated using the built-in image generation tool; its exact prompt is saved in `assets/rural-perception-flow-prompt.txt`.
