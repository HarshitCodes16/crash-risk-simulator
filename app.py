"""
Crash Risk Simulator - Streamlit App
--------------------------------------
A what-if scenario simulator (NOT a live telemetry tracker). The user sets a
hypothetical driving scenario via 4 real-world dropdowns + 2 direct sliders,
and the app predicts crash risk with an independent physics-based verification
layer, plus actionable recommendations.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import plotly.graph_objects as go

from simulation.dependency_map import (
    BASE_RANGES, WEATHER_OPTIONS, ROAD_OPTIONS, TIME_OPTIONS, VEHICLE_OPTIONS,
    derived_params_midpoint,
)
from physics.crash_physics import verify_against_ml
from models.predict import (
    load_models, predict_risk, sensitivity_sweep, biggest_risk_factor,
    threshold_crossing, FEATURES,
)

st.set_page_config(page_title="Crash Risk Simulator", page_icon="🚗", layout="centered")

FEATURE_LABELS = {
    "speed": "Speed",
    "distance": "Following distance",
    "reaction_time": "Reaction time",
    "brake_eff": "Brake efficiency",
    "friction": "Road friction",
    "mass": "Vehicle mass",
}

RISK_THRESHOLD = 0.5

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


@st.cache_resource
def get_models():
    """
    Loads the saved models. If they fail to load (e.g. a scikit-learn/Python
    version mismatch between the machine that trained them and the machine
    running the app - which is exactly what happens across different
    deployment environments), fall back to training fresh, in-memory, using
    whatever library versions are installed here. This makes the app
    resilient to environment differences instead of crashing on deploy.
    """
    try:
        return load_models()
    except Exception:
        st.info("Saved models weren't compatible with this environment — training fresh models now (takes a few seconds)...")
        from simulation.generate_dataset import generate_dataset
        from models.train import train_stage1_classifier, train_stage2_regressor

        df = generate_dataset(n_samples=4000, seed=42)
        clf, _, _ = train_stage1_classifier(df)
        reg, _, _ = train_stage2_regressor(df)
        return clf, reg


def main():
    st.title("🚗 Crash Risk Simulator")
    st.caption(
        "A what-if scenario tool — not a live tracker. Set a hypothetical driving "
        "scenario below and see the predicted crash risk, with an independent "
        "physics check on the model's output."
    )

    try:
        clf, reg = get_models()
    except FileNotFoundError:
        st.error(
            "No trained models found. Run `python src/simulation/generate_dataset.py` "
            "then `python src/models/train.py` first to generate data and train the models."
        )
        return

    st.subheader("Scenario inputs")

    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("🌦️ Weather", WEATHER_OPTIONS, index=0)
        time_of_day = st.selectbox("🌙 Time of day", TIME_OPTIONS, index=0)
    with col2:
        road_type = st.selectbox("🛣️ Road type", ROAD_OPTIONS, index=1)
        vehicle_type = st.selectbox("🚙 Vehicle type", VEHICLE_OPTIONS, index=1)

    speed = st.slider("Speed (m/s)", float(BASE_RANGES["speed"][0]), float(BASE_RANGES["speed"][1]), 25.0)
    distance = st.slider("Following distance (m)", float(BASE_RANGES["distance"][0]), float(BASE_RANGES["distance"][1]), 20.0)

    # Derive the 4 hidden physics parameters from the categorical choices
    derived = derived_params_midpoint(weather, road_type, time_of_day, vehicle_type)
    reaction_time = derived["reaction_time"]
    friction = derived["friction"]
    mass = derived["mass"]
    brake_eff = derived["brake_eff"]

    crash_probability, severity = predict_risk(
        clf, reg, speed, distance, reaction_time, brake_eff, friction, mass
    )

    st.divider()
    st.subheader("Risk assessment")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Crash probability", f"{crash_probability * 100:.0f}%")
    with c2:
        st.metric("Predicted severity", f"{severity:,.0f} N" if crash_probability >= RISK_THRESHOLD else "—")

    # --- Verification layer ---
    verification = verify_against_ml(
        speed, distance, reaction_time, brake_eff, friction, mass,
        crash_probability, probability_threshold=RISK_THRESHOLD,
    )
    if verification["status"] == "agree":
        st.success("✅ ML prediction and physics check agree — high confidence.")
    elif verification["status"] == "uncertain":
        st.warning(
            "🟡 This scenario is borderline — the crash probability sits right near "
            "the decision threshold, where the ML model and the physics check can "
            "reasonably differ. Treat this as a moderate-risk, low-confidence case "
            "rather than a clear-cut answer."
        )
    else:
        st.error(
            "⚠️ ML prediction and the physics-based stopping-distance check clearly "
            "disagree for this scenario — treat this result with extra caution."
        )

    # --- Danger threshold warning ---
    if crash_probability >= RISK_THRESHOLD:
        st.error("🚨 Unsafe distance for this speed.")

    # --- Biggest risk factor ---
    fixed_values = {
        "speed": speed, "distance": distance, "reaction_time": reaction_time,
        "brake_eff": brake_eff, "friction": friction, "mass": mass,
    }
    param_ranges = {
        "speed": BASE_RANGES["speed"],
        "distance": BASE_RANGES["distance"],
        "reaction_time": (0.3, 5.0),
        "brake_eff": (0.1, 1.0),
        "friction": (0.1, 1.0),
        "mass": (700, 3500),
    }
    biggest, swings = biggest_risk_factor(clf, fixed_values, param_ranges)
    st.write(f"⚡ **Biggest risk factor right now:** {FEATURE_LABELS[biggest]}")

    # --- Safe speed recommendation ---
    speed_result = threshold_crossing(clf, "speed", BASE_RANGES["speed"], fixed_values, threshold=RISK_THRESHOLD)
    safe_speed = speed_result["safe_value"]
    if safe_speed is not None:
        st.info(f"🛡️ Keep speed under **{safe_speed:.0f} m/s** to stay below {RISK_THRESHOLD*100:.0f}% risk in these conditions.")
    else:
        st.error("🛡️ Even the minimum speed is unsafe in these conditions — consider not driving until conditions improve.")

    # --- Safe distance recommendation (the reverse: for this speed, what gap do I need?) ---
    distance_result = threshold_crossing(clf, "distance", BASE_RANGES["distance"], fixed_values, threshold=RISK_THRESHOLD)
    safe_distance = distance_result["safe_value"]
    if safe_distance is not None:
        st.info(f"🛡️ Keep at least **{safe_distance:.0f} m** of following distance to stay below {RISK_THRESHOLD*100:.0f}% risk at this speed.")
    else:
        st.error("🛡️ No following distance in the available range keeps this speed safe — slow down.")

    def _sweep_chart(title, x_label, result, current_x, current_prob):
        x_values, probs = result["x_values"], result["probabilities"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, y=probs * 100, mode="lines", fill="tozeroy",
            line=dict(color="#D85A30", width=2), fillcolor="rgba(216,90,48,0.15)",
            name="Crash probability",
        ))
        fig.add_hline(y=RISK_THRESHOLD * 100, line_dash="dash", line_color="gray",
                      annotation_text=f"{RISK_THRESHOLD*100:.0f}% threshold")
        # Mark the safe-value threshold crossing directly on the chart
        if result["safe_value"] is not None:
            fig.add_vline(
                x=result["safe_value"], line_dash="dot", line_color="#2E7D32",
                annotation_text=f"safe boundary: {result['safe_value']:.1f}",
                annotation_position="top",
            )
        fig.add_trace(go.Scatter(
            x=[current_x], y=[current_prob * 100], mode="markers+text",
            marker=dict(size=10, color="#D85A30"), text=["you are here"],
            textposition="bottom center", showlegend=False,
        ))
        fig.update_layout(
            title=title, xaxis_title=x_label, yaxis_title="Crash probability (%)",
            height=340, margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    # --- Sensitivity sweep charts: Speed and Distance side by side ---
    st.divider()
    st.subheader("Risk sweep charts")

    tab1, tab2, tab3 = st.tabs(["Risk vs. Speed", "Risk vs. Distance", "Explore any factor"])

    with tab1:
        fig_speed = _sweep_chart(
            "Risk vs. Speed (distance held fixed)", "Speed (m/s)",
            speed_result, speed, crash_probability,
        )
        st.plotly_chart(fig_speed, width='stretch', config=PLOTLY_CONFIG)

    with tab2:
        fig_distance = _sweep_chart(
            "Risk vs. Distance (speed held fixed)", "Following distance (m)",
            distance_result, distance, crash_probability,
        )
        st.plotly_chart(fig_distance, width='stretch', config=PLOTLY_CONFIG)

    with tab3:
        st.caption("Pick any of the 6 underlying factors to see how risk changes across its full range, with the current scenario and safe boundary marked.")
        explore_param = st.selectbox(
            "Factor to explore",
            options=FEATURES,
            format_func=lambda f: FEATURE_LABELS[f],
        )
        explore_result = threshold_crossing(clf, explore_param, param_ranges[explore_param], fixed_values, threshold=RISK_THRESHOLD)
        fig_explore = _sweep_chart(
            f"Risk vs. {FEATURE_LABELS[explore_param]}", FEATURE_LABELS[explore_param],
            explore_result, fixed_values[explore_param], crash_probability,
        )
        st.plotly_chart(fig_explore, width='stretch', config=PLOTLY_CONFIG)

    with st.expander("See sensitivity breakdown for all factors"):
        for param, swing in sorted(swings.items(), key=lambda x: -x[1]):
            st.write(f"- **{FEATURE_LABELS[param]}**: {swing*100:.1f} percentage-point swing")


if __name__ == "__main__":
    main()
