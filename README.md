# 🚗 Crash Risk Simulator

A two-stage ML system for predictive driving safety assessment — an interactive
Streamlit app that predicts crash risk (with a confidence score) and severity
for a given driving scenario, cross-checked by an independent physics layer.

> **This is a what-if scenario simulator, not a live telemetry tracker.** It
> doesn't track a real moving car — the user sets a hypothetical scenario
> ("if I were driving at 32 m/s, 15m behind the car ahead, in the rain, at
> night, in a truck — how risky is that?") and the app predicts the risk.
> The architecture is designed so this manual input layer can later be
> replaced with live sensor/GPS/weather-API feeds (V2 vision) without
> touching the core ML pipeline.

---

## The problem

Road accidents are rarely caused by one factor alone — they're multiple
conditions stacking up (speed, gap, weather, visibility, vehicle type). Real
crash data is dangerous, expensive, and ethically impossible to collect at
scale, so this project generates realistic scenario data through **simulation**
instead, and builds a system that doesn't just predict "how hard would the
impact be" but answers the more useful question: **"given how I'm driving
right now, how risky is this — and what should I change?"**

## Architecture

```
User inputs
 ├── Speed (slider)              ─┐
 ├── Distance (slider)            ├──► 6 simulation parameters
 ├── Weather (dropdown)           │     (speed, distance, reaction_time,
 ├── Road type (dropdown)         │      brake_eff, friction, mass)
 ├── Time of day (dropdown)       │
 └── Vehicle type (dropdown)     ─┘
        │
        ▼
 Dependency mapping layer (multiplicative, range-based, clamped)
   weather + road type      → friction
   weather + time of day    → reaction_time
   vehicle type              → mass, brake_eff
        │
        ▼
 Stage 1: Crash Classifier (best of Logistic/DecisionTree/RandomForest/GradientBoosting)
        │  ──► crash probability (%)
        ▼
 Stage 2: Severity Regressor (trained only on crash-positive rows)
        │  ──► impact force
        ▼
 Independent physics verification layer
   stopping_distance = v·t_reaction + v² / (2·μ·η·g)
   flags disagreement between ML output and classical physics
        │
        ▼
 Streamlit UI: probability, severity, verification badge, danger warning,
               biggest-risk-factor callout, safe-speed recommendation,
               sensitivity sweep chart
```

## Features

- **Two-stage prediction** — crash probability, then conditional severity
- **Real-world categorical inputs** mapped internally to physics parameters
- **Independent physics verification layer** (generator + verifier pattern)
- **Danger threshold warning** when risk crosses 50%
- **Biggest risk factor callout** — which input is driving risk the most
- **Safe speed recommendation** — reverse-calculated max safe speed
- **Sensitivity sweep chart** with threshold line and risk gradient (Plotly)
- **Model comparison** — 4 algorithms benchmarked per stage, best one kept

## Project structure

```
crash-risk-simulator/
├── app.py                       # Streamlit UI
├── requirements.txt
├── data/
│   └── crash_simulation_data_v2.csv
├── saved_models/
│   ├── stage1_classifier.pkl
│   └── stage2_regressor.pkl
├── results/
│   ├── metrics_summary.json     # full model comparison + metrics
│   ├── stage1_confusion_matrix.png
│   └── *_feature_importance.png
└── src/
    ├── simulation/
    │   ├── dependency_map.py    # categorical -> physics parameter mapping
    │   └── generate_dataset.py  # SimPy-based data generator
    ├── physics/
    │   └── crash_physics.py     # stopping-distance formula + verification
    └── models/
        ├── train.py             # trains + compares + saves both stages
        └── predict.py           # prediction, sensitivity, safe-speed logic
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the simulated dataset
python src/simulation/generate_dataset.py

# 2. Train both stages (saves .pkl models + metrics/plots)
python src/models/train.py

# 3. Launch the app
streamlit run app.py
```

## Model performance (current run)

**Stage 1 — Crash Classifier** (best: Random Forest, ROC-AUC ≈ 0.996)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RandomForest | 0.965 | 0.975 | 0.975 | 0.975 | 0.996 |
| GradientBoosting | 0.960 | 0.978 | 0.965 | 0.971 | 0.996 |
| LogisticRegression | 0.956 | 0.977 | 0.961 | 0.969 | 0.992 |
| DecisionTree | 0.919 | 0.953 | 0.931 | 0.942 | 0.910 |

**Stage 2 — Severity Regressor** (best: Gradient Boosting, R² ≈ 0.96)
| Model | R² | RMSE | MAE |
|---|---|---|---|
| GradientBoosting | 0.960 | 179,491 | 118,089 |
| RandomForest | 0.951 | 197,973 | 124,126 |
| DecisionTree | 0.868 | 326,145 | 203,177 |
| LinearRegression | 0.847 | 350,735 | 257,557 |

Full metrics saved in `results/metrics_summary.json`.

## Interview talking points

- Reframed the problem from pure regression to a two-stage pipeline
  (probability of occurrence → conditional severity), similar to how
  real-world risk assessment systems (insurance, safety) are structured.
- Modeled weather and traffic as multi-variable dependencies — e.g. rain
  reduces both friction *and* effective reaction time (visibility), rather
  than a single relabeled slider.
- Verified the ML model's output against an independent, deterministic
  physics formula rather than trusting it blindly — same generator/verifier
  philosophy applied elsewhere, just rule-based here instead of a second
  neural model.
- Speed and vehicle mass dominate the risk model, which lines up with
  kinetic energy scaling as mass × velocity² — a physics-grounded sanity
  check on the model's behavior.

## Future scope (V2, not built)

Replace manual sliders/dropdowns with live sensor input: speedometer for
speed, GPS/radar for following distance, a weather API for conditions — to
turn this into a real in-vehicle safety alert system (ADAS-style). The input
layer is deliberately decoupled from the model/physics pipeline so this swap
wouldn't require redesigning the core system.
