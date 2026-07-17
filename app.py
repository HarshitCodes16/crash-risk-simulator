"""
Crash Risk Simulator - Streamlit App
--------------------------------------
A what-if scenario simulator (NOT a live telemetry tracker). Home page lets
the user pick between scenarios, explore model evaluation, or try quiz mode.
"""

import os
import sys
import json
import random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from simulation.dependency_map import (
    BASE_RANGES, WEATHER_OPTIONS, ROAD_OPTIONS, TIME_OPTIONS, VEHICLE_OPTIONS,
    derived_params_midpoint,
)
from physics.crash_physics import verify_against_ml
from models.predict import (
    load_models, predict_risk, sensitivity_sweep, biggest_risk_factor,
    threshold_crossing, FEATURES,
)
from utils.weather import (
    geocode_city_suggestions, fetch_current_conditions, weather_code_to_category, local_hour_to_time_of_day,
)
from utils.explain import explain_prediction

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
MS_TO_KMH = 3.6

def ms_to_kmh(v):
    return v * MS_TO_KMH

def kmh_to_ms(v):
    return v / MS_TO_KMH

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

PRESETS_EB = {
    "🌧️ Heavy Rain": {"weather": "Rainy", "road": "City", "time": "Day", "vehicle": "Sedan", "speed_kmh": 60.0, "distance": 15.0},
    "🌫️ Fog": {"weather": "Foggy", "road": "Highway", "time": "Day", "vehicle": "Sedan", "speed_kmh": 70.0, "distance": 20.0},
    "🌙 Night Highway": {"weather": "Sunny", "road": "Highway", "time": "Night", "vehicle": "Sedan", "speed_kmh": 110.0, "distance": 40.0},
    "🚦 Traffic Jam": {"weather": "Sunny", "road": "City", "time": "Day", "vehicle": "Hatchback", "speed_kmh": 15.0, "distance": 5.0},
    "🏫 School Zone": {"weather": "Sunny", "road": "City", "time": "Day", "vehicle": "Sedan", "speed_kmh": 25.0, "distance": 10.0},
}

PRESETS_DT = {
    "🌧️ Heavy Rain": {"weather": "Rainy", "road": "City", "time": "Day", "vehicle": "Sedan", "your_speed_kmh": 70.0, "traffic_speed_kmh": 55.0, "distance": 15.0},
    "🌫️ Fog": {"weather": "Foggy", "road": "Highway", "time": "Day", "vehicle": "Sedan", "your_speed_kmh": 90.0, "traffic_speed_kmh": 75.0, "distance": 25.0},
    "🌙 Night Highway": {"weather": "Sunny", "road": "Highway", "time": "Night", "vehicle": "Sedan", "your_speed_kmh": 120.0, "traffic_speed_kmh": 100.0, "distance": 35.0},
    "🚦 Traffic Jam": {"weather": "Sunny", "road": "City", "time": "Day", "vehicle": "Hatchback", "your_speed_kmh": 20.0, "traffic_speed_kmh": 8.0, "distance": 6.0},
    "🏫 School Zone": {"weather": "Sunny", "road": "City", "time": "Day", "vehicle": "Sedan", "your_speed_kmh": 35.0, "traffic_speed_kmh": 25.0, "distance": 10.0},
}


@st.cache_resource
def get_models():
    """
    Loads the saved models. If they fail to load (e.g. a scikit-learn/Python
    version mismatch between the machine that trained them and the machine
    running the app), fall back to training fresh, in-memory, using whatever
    library versions are installed here.

    IMPORTANT: this function must never call a Streamlit UI element (st.*)
    directly - Streamlit's cache "replay" mechanism re-executes any UI calls
    made inside a @st.cache_resource function on every cache hit, tied to a
    specific layout block. On a fresh script run after a redeploy/reboot,
    that original block no longer exists, which raises a hard
    CacheReplayClosureError and crashes the whole app. Any user-facing
    feedback about what happened here must be shown OUTSIDE this function,
    driven by its return value instead.
    """
    try:
        clf, reg = load_models()
        return clf, reg, False
    except Exception:
        from simulation.generate_dataset import generate_dataset
        from models.train import train_fast_fallback

        df = generate_dataset(n_samples=1500, seed=42)
        clf, reg = train_fast_fallback(df)
        return clf, reg, True


@st.cache_data
def get_background_sample():
    """
    A small sample of scenario data used as SHAP's reference background
    distribution. Tries the saved dataset first; falls back to generating a
    small fresh one so this never blocks the app if the CSV isn't present.
    """
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "crash_simulation_data_v2.csv"))
        return df[FEATURES].sample(min(40, len(df)), random_state=42)
    except Exception:
        from simulation.generate_dataset import generate_dataset
        df = generate_dataset(n_samples=200, seed=1)
        return df[FEATURES].sample(min(40, len(df)), random_state=42)


def _sweep_chart(title, x_label, result, current_x, current_prob, is_speed=False):
    x_values, probs = result["x_values"], result["probabilities"]
    display_x = ms_to_kmh(x_values) if is_speed else x_values
    display_current_x = ms_to_kmh(current_x) if is_speed else current_x
    display_safe = ms_to_kmh(result["safe_value"]) if (is_speed and result["safe_value"] is not None) else result["safe_value"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_x, y=probs * 100, mode="lines", fill="tozeroy",
        line=dict(color="#D85A30", width=2), fillcolor="rgba(216,90,48,0.15)",
        name="Crash probability",
    ))
    fig.add_hline(y=RISK_THRESHOLD * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"{RISK_THRESHOLD*100:.0f}% threshold")
    if display_safe is not None:
        fig.add_vline(
            x=display_safe, line_dash="dot", line_color="#2E7D32",
            annotation_text=f"safe boundary: {display_safe:.1f}",
            annotation_position="top",
        )
    fig.add_trace(go.Scatter(
        x=[display_current_x], y=[current_prob * 100], mode="markers+text",
        marker=dict(size=10, color="#D85A30"), text=["you are here"],
        textposition="bottom center", showlegend=False,
    ))
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title="Crash probability (%)",
        height=340, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _swings_chart_png(swings, speed_label):
    """Static (non-interactive) version of the risk-contribution bar chart,
    rendered with matplotlib specifically so it can be embedded as an image
    in the PDF export (Plotly charts need the heavy 'kaleido' package to
    export as static images, which we're deliberately avoiding here)."""
    labels, values = [], []
    for param, swing in sorted(swings.items(), key=lambda x: x[1]):
        labels.append(FEATURE_LABELS[param] if param != "speed" else speed_label)
        values.append(swing * 100)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, values, color="#D85A30")
    ax.set_xlabel("Percentage-point swing")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _risk_bar_chart(swings, speed_label):
    labels = []
    values = []
    for param, swing in sorted(swings.items(), key=lambda x: x[1]):
        labels.append(FEATURE_LABELS[param] if param != "speed" else speed_label)
        values.append(swing * 100)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color="#D85A30"),
        text=[f"{v:.1f} pts" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Risk contribution — how much each factor swings crash probability",
        xaxis_title="Percentage-point swing", height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _param_ranges():
    return {
        "speed": BASE_RANGES["speed"],
        "distance": BASE_RANGES["distance"],
        "reaction_time": (0.3, 5.0),
        "brake_eff": (0.1, 1.0),
        "friction": (0.1, 1.0),
        "mass": (700, 3500),
    }


def _apply_preset(preset, prefix, is_dynamic):
    st.session_state[f"{prefix}_weather"] = preset["weather"]
    st.session_state[f"{prefix}_road"] = preset["road"]
    st.session_state[f"{prefix}_time"] = preset["time"]
    st.session_state[f"{prefix}_vehicle"] = preset["vehicle"]
    if is_dynamic:
        st.session_state[f"{prefix}_your_speed"] = preset["your_speed_kmh"]
        st.session_state[f"{prefix}_traffic_speed"] = preset["traffic_speed_kmh"]
    else:
        st.session_state[f"{prefix}_speed"] = preset["speed_kmh"]
    st.session_state[f"{prefix}_distance"] = preset["distance"]


def _render_presets(presets, prefix, is_dynamic):
    st.caption("Quick presets:")
    cols = st.columns(len(presets))
    for col, (name, preset) in zip(cols, presets.items()):
        with col:
            if st.button(name, key=f"{prefix}_preset_{name}", width='stretch'):
                _apply_preset(preset, prefix, is_dynamic)
                st.rerun()


def _render_weather_autofill(prefix):
    with st.expander("📍 Auto-fill weather & time from a city (optional)"):
        city = st.text_input("City name", key=f"{prefix}_city_input", placeholder="e.g. Mumbai")
        search_clicked = st.button("🔎 Search city", key=f"{prefix}_city_search")

        if search_clicked and city.strip():
            with st.spinner("Searching cities..."):
                st.session_state[f"{prefix}_city_results"] = geocode_city_suggestions(city.strip(), count=5)
                st.session_state[f"{prefix}_city_searched_for"] = city.strip()

        suggestions = st.session_state.get(f"{prefix}_city_results", [])
        searched_for = st.session_state.get(f"{prefix}_city_searched_for", "")

        selected_geo = None
        if suggestions:
            labels = [
                f"{s['name']}" + (f", {s['admin1']}" if s['admin1'] else "") + f", {s['country']}"
                for s in suggestions
            ]
            # Key includes the searched text itself, so a new search always
            # creates a fresh dropdown (defaulting back to the top match)
            # instead of keeping a previous search's selected index.
            choice_idx = st.selectbox(
                "Did you mean:", options=list(range(len(labels))),
                format_func=lambda i: labels[i], key=f"{prefix}_city_choice_{searched_for}",
            )
            selected_geo = suggestions[choice_idx]
        elif search_clicked:
            st.caption("No matching cities found — check the spelling and search again.")

        if st.button("Fetch current weather", key=f"{prefix}_fetch_weather", disabled=selected_geo is None):
            with st.spinner("Fetching weather..."):
                conditions = fetch_current_conditions(selected_geo["lat"], selected_geo["lon"])
            if conditions is None:
                st.error("Couldn't fetch weather right now — please select manually below.")
            else:
                category = weather_code_to_category(conditions["weather_code"])
                time_cat = local_hour_to_time_of_day(conditions["local_time"])
                st.session_state[f"{prefix}_weather"] = category
                st.session_state[f"{prefix}_time"] = time_cat
                temp = conditions["temperature"]
                st.success(
                    f"Detected **{category}**, **{time_cat}** in {selected_geo['name']}, "
                    f"{selected_geo['country']} ({temp}°C) — updated below."
                )


def render_risk_assessment(clf, reg, mode_name, speed_label, speed_value, distance, weather, road_type, time_of_day, vehicle_type):
    """
    Shared risk-assessment block used by BOTH scenarios. `speed_value` is
    whatever quantity actually represents the closing/approach speed for that
    scenario - the car's own speed for Emergency Braking, or the relative
    speed (user speed - traffic ahead speed) for Dynamic Traffic.
    """
    derived = derived_params_midpoint(weather, road_type, time_of_day, vehicle_type)
    reaction_time = derived["reaction_time"]
    friction = derived["friction"]
    mass = derived["mass"]
    brake_eff = derived["brake_eff"]

    if speed_value <= 0:
        st.success("✅ You're not closing the gap — no collision risk in this scenario.")
        return

    crash_probability, severity = predict_risk(
        clf, reg, speed_value, distance, reaction_time, brake_eff, friction, mass
    )

    st.divider()
    st.subheader("Risk assessment")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Crash probability", f"{crash_probability * 100:.0f}%")
    with c2:
        st.metric("Predicted severity", f"{severity:,.0f} N" if crash_probability >= RISK_THRESHOLD else "—")

    verification = verify_against_ml(
        speed_value, distance, reaction_time, brake_eff, friction, mass,
        crash_probability, probability_threshold=RISK_THRESHOLD,
    )
    if verification["status"] == "agree":
        st.success("✅ ML prediction and physics check agree — high confidence.")
    elif verification["status"] == "uncertain":
        st.warning(
            "🟡 This scenario is borderline — the crash probability sits right near "
            "the decision threshold, where the ML model and the physics check can "
            "reasonably differ. Treat this as a moderate-risk, low-confidence case."
        )
    else:
        st.error(
            "⚠️ ML prediction and the physics-based stopping-distance check clearly "
            "disagree for this scenario — treat this result with extra caution."
        )

    if crash_probability >= RISK_THRESHOLD:
        st.error("🚨 Unsafe distance for this speed.")

    fixed_values = {
        "speed": speed_value, "distance": distance, "reaction_time": reaction_time,
        "brake_eff": brake_eff, "friction": friction, "mass": mass,
    }
    param_ranges = _param_ranges()
    biggest, swings = biggest_risk_factor(clf, fixed_values, param_ranges)
    st.write(f"⚡ **Biggest risk factor right now:** {FEATURE_LABELS[biggest] if biggest != 'speed' else speed_label}")

    st.plotly_chart(_risk_bar_chart(swings, speed_label), width='stretch', config=PLOTLY_CONFIG)

    speed_result = threshold_crossing(clf, "speed", BASE_RANGES["speed"], fixed_values, threshold=RISK_THRESHOLD)
    safe_speed = speed_result["safe_value"]
    distance_result = threshold_crossing(clf, "distance", BASE_RANGES["distance"], fixed_values, threshold=RISK_THRESHOLD)
    safe_distance = distance_result["safe_value"]

    # --- Recommendation engine ---
    recommendations_list = []
    if crash_probability >= RISK_THRESHOLD:
        st.subheader("💡 Recommendations")
        if safe_speed is not None and safe_speed < speed_value:
            prob_at_safe = float(np.interp(safe_speed, speed_result["x_values"], speed_result["probabilities"]))
            delta_kmh = ms_to_kmh(speed_value - safe_speed)
            rec_text = (
                f"Reduce {speed_label.lower()} by {delta_kmh:.0f} km/h "
                f"(to {ms_to_kmh(safe_speed):.0f} km/h) — crash probability drops "
                f"from {crash_probability*100:.0f}% to about {prob_at_safe*100:.0f}%."
            )
            recommendations_list.append(rec_text)
            st.info(f"**{rec_text}**")
        if safe_distance is not None and safe_distance > distance:
            prob_at_safe_d = float(np.interp(safe_distance, distance_result["x_values"], distance_result["probabilities"]))
            delta_m = safe_distance - distance
            rec_text = (
                f"Increase following distance by {delta_m:.0f} m "
                f"(to {safe_distance:.0f} m) — crash probability drops "
                f"from {crash_probability*100:.0f}% to about {prob_at_safe_d*100:.0f}%."
            )
            recommendations_list.append(rec_text)
            st.info(f"**{rec_text}**")
        if safe_speed is None and safe_distance is None:
            st.error("No single adjustment within the available range makes this scenario safe — conditions are too severe.")
    else:
        st.success(f"✅ No changes needed — this scenario is already under the {RISK_THRESHOLD*100:.0f}% risk threshold.")

    if safe_speed is not None:
        st.info(f"🛡️ Keep {speed_label.lower()} under **{ms_to_kmh(safe_speed):.0f} km/h** to stay below {RISK_THRESHOLD*100:.0f}% risk in these conditions.")
    else:
        st.error(f"🛡️ Even the minimum {speed_label.lower()} is unsafe in these conditions.")

    if safe_distance is not None:
        st.info(f"🛡️ Keep at least **{safe_distance:.0f} m** of following distance to stay below {RISK_THRESHOLD*100:.0f}% risk.")
    else:
        st.error("🛡️ No following distance in the available range keeps this scenario safe — slow down.")

    st.divider()
    st.subheader("Risk sweep charts")
    tab1, tab2, tab3 = st.tabs([f"Risk vs. {speed_label}", "Risk vs. Distance", "Explore any factor"])

    with tab1:
        fig_speed = _sweep_chart(
            f"Risk vs. {speed_label} (distance held fixed)", f"{speed_label} (km/h)",
            speed_result, speed_value, crash_probability, is_speed=True,
        )
        st.plotly_chart(fig_speed, width='stretch', config=PLOTLY_CONFIG)

    with tab2:
        fig_distance = _sweep_chart(
            "Risk vs. Distance (speed held fixed)", "Following distance (m)",
            distance_result, distance, crash_probability,
        )
        st.plotly_chart(fig_distance, width='stretch', config=PLOTLY_CONFIG)

    with tab3:
        st.caption("Pick any of the 6 underlying factors to see how risk changes across its full range.")
        explore_param = st.selectbox(
            "Factor to explore", options=FEATURES,
            format_func=lambda f: FEATURE_LABELS[f] if f != "speed" else speed_label,
            key=f"explore_{speed_label}",
        )
        explore_result = threshold_crossing(clf, explore_param, param_ranges[explore_param], fixed_values, threshold=RISK_THRESHOLD)
        explore_is_speed = explore_param == "speed"
        explore_unit = "km/h" if explore_is_speed else ""
        explore_label = FEATURE_LABELS[explore_param] if explore_param != "speed" else speed_label
        fig_explore = _sweep_chart(
            f"Risk vs. {explore_label}",
            f"{explore_label} ({explore_unit})" if explore_unit else explore_label,
            explore_result, fixed_values[explore_param], crash_probability, is_speed=explore_is_speed,
        )
        st.plotly_chart(fig_explore, width='stretch', config=PLOTLY_CONFIG)

    # --- Per-scenario SHAP explanation (on-demand, since it can be slow for some model types) ---
    st.divider()
    st.subheader("🔍 Explain this specific prediction")
    st.caption(
        "The risk contribution bars above show what generally matters across "
        "the full range. This instead explains *this exact scenario*: how much "
        "did each factor push today's prediction up or down from the model's "
        "average baseline?"
    )
    if st.button("Compute explanation", key=f"shap_button_{speed_label}"):
        with st.spinner("Computing per-scenario explanation (can take a few seconds)..."):
            background_df = get_background_sample()
            explanation = explain_prediction(clf, background_df, pd.DataFrame([fixed_values])[FEATURES])
        if explanation is None:
            st.caption("A per-scenario explanation isn't available right now for this model.")
        else:
            labels, values = [], []
            for param, val in sorted(explanation["values"].items(), key=lambda x: x[1]):
                labels.append(FEATURE_LABELS[param] if param != "speed" else speed_label)
                values.append(val * 100)
            colors = ["#2E7D32" if v < 0 else "#D85A30" for v in values]
            fig_shap = go.Figure(go.Bar(
                x=values, y=labels, orientation="h", marker=dict(color=colors),
                text=[f"{v:+.1f} pts" for v in values], textposition="outside",
            ))
            fig_shap.update_layout(
                title=f"Baseline {explanation['base']*100:.0f}% → this scenario {crash_probability*100:.0f}%",
                xaxis_title="Contribution to crash probability (percentage points)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_shap, width='stretch', config=PLOTLY_CONFIG)
            st.caption("Green bars pushed risk down for this scenario; orange bars pushed it up.")

    st.divider()
    verification_labels = {
        "agree": "ML and physics agree - high confidence",
        "uncertain": "Borderline - near decision threshold",
        "disagree": "ML and physics disagree - use caution",
    }
    pdf_scenario = {
        "mode": mode_name, "speed_label": speed_label, "speed_kmh": ms_to_kmh(speed_value),
        "distance": distance, "weather": weather, "road_type": road_type,
        "time_of_day": time_of_day, "vehicle_type": vehicle_type,
    }
    pdf_results = {
        "crash_probability": crash_probability,
        "severity": severity if crash_probability >= RISK_THRESHOLD else None,
        "verification_status": verification_labels[verification["status"]],
        "biggest_factor": FEATURE_LABELS[biggest] if biggest != "speed" else speed_label,
        "recommendations": recommendations_list,
        "safe_speed_kmh": ms_to_kmh(safe_speed) if safe_speed is not None else None,
        "safe_distance": safe_distance,
        "chart_image_bytes": _swings_chart_png(swings, speed_label),
    }
    try:
        from utils.pdf_report import generate_scenario_pdf
        pdf_bytes = generate_scenario_pdf(pdf_scenario, pdf_results)
        st.download_button(
            "📄 Download this scenario as PDF", data=pdf_bytes,
            file_name=f"crash_risk_report_{mode_name.lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.caption(f"PDF export isn't available right now ({e}).")


def render_home():
    st.title("🚗 Crash Risk Simulator")
    st.caption("Simulation-driven driving safety assessment — choose a scenario to explore.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🚨 Emergency Braking")
            st.write(
                "The vehicle ahead suddenly slams the brakes. Given your speed, "
                "following distance, and conditions — what's your crash risk?"
            )
            if st.button("Launch Emergency Braking →", width='stretch', type="primary"):
                st.session_state.view = "emergency"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("🚗 Dynamic Traffic")
            st.write(
                "Both vehicles are moving. Risk depends on the *relative* speed "
                "between you and the traffic ahead — closer to real highway driving."
            )
            if st.button("Launch Dynamic Traffic →", width='stretch', type="primary"):
                st.session_state.view = "dynamic"
                st.rerun()

    st.write("")
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.subheader("🎯 Quiz Mode")
            st.write("Guess the crash risk before the model reveals it. Test your driving-safety intuition.")
            if st.button("Play Quiz →", width='stretch'):
                st.session_state.view = "quiz"
                st.rerun()
    with col4:
        with st.container(border=True):
            st.subheader("📊 Model Evaluation")
            st.write("See how the underlying models were trained, compared, and evaluated.")
            if st.button("View Dashboard →", width='stretch'):
                st.session_state.view = "evaluation"
                st.rerun()


def render_emergency_braking(clf, reg):
    if st.button("← Back to home"):
        st.session_state.view = "home"
        st.rerun()

    st.title("🚨 Emergency Braking Simulator")
    st.caption(
        "A what-if scenario tool — not a live tracker. The car ahead suddenly "
        "brakes. Set a hypothetical scenario and see the predicted crash risk."
    )

    _render_presets(PRESETS_EB, "eb", is_dynamic=False)
    _render_weather_autofill("eb")

    st.subheader("Scenario inputs")
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("🌦️ Weather", WEATHER_OPTIONS, key="eb_weather")
        time_of_day = st.selectbox("🌙 Time of day", TIME_OPTIONS, key="eb_time")
    with col2:
        road_type = st.selectbox("🛣️ Road type", ROAD_OPTIONS, key="eb_road")
        vehicle_type = st.selectbox("🚙 Vehicle type", VEHICLE_OPTIONS, key="eb_vehicle")

    st.session_state.setdefault("eb_speed", 90.0)
    st.session_state.setdefault("eb_distance", 20.0)
    speed_kmh = st.slider("Speed (km/h)", float(ms_to_kmh(BASE_RANGES["speed"][0])), float(ms_to_kmh(BASE_RANGES["speed"][1])), key="eb_speed")
    speed = kmh_to_ms(speed_kmh)
    distance = st.slider("Following distance (m)", float(BASE_RANGES["distance"][0]), float(BASE_RANGES["distance"][1]), key="eb_distance")

    render_risk_assessment(clf, reg, "Emergency Braking", "Speed", speed, distance, weather, road_type, time_of_day, vehicle_type)


def render_dynamic_traffic(clf, reg):
    if st.button("← Back to home"):
        st.session_state.view = "home"
        st.rerun()

    st.title("🚗 Dynamic Traffic Simulator")
    st.caption(
        "Both vehicles are moving. What matters isn't your absolute speed, but "
        "how much faster you're closing in on the traffic ahead."
    )
    st.info(
        "ℹ️ This models a **steady closing speed** — e.g. you're gradually "
        "gaining on slower traffic. It assumes the gap keeps closing at this "
        "rate; it does **not** model the traffic ahead suddenly slamming its "
        "brakes to a full stop. For that sudden-full-stop risk, use the "
        "**Emergency Braking** scenario instead, with your actual speed."
    )

    _render_presets(PRESETS_DT, "dt", is_dynamic=True)
    _render_weather_autofill("dt")

    st.subheader("Scenario inputs")
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("🌦️ Weather", WEATHER_OPTIONS, key="dt_weather")
        time_of_day = st.selectbox("🌙 Time of day", TIME_OPTIONS, key="dt_time")
    with col2:
        road_type = st.selectbox("🛣️ Road type", ROAD_OPTIONS, key="dt_road")
        vehicle_type = st.selectbox("🚙 Vehicle type", VEHICLE_OPTIONS, key="dt_vehicle")

    st.session_state.setdefault("dt_your_speed", 108.0)
    st.session_state.setdefault("dt_traffic_speed", 100.0)
    st.session_state.setdefault("dt_distance", 20.0)
    your_speed_kmh = st.slider("Your speed (km/h)", float(ms_to_kmh(BASE_RANGES["speed"][0])), float(ms_to_kmh(BASE_RANGES["speed"][1])), key="dt_your_speed")
    traffic_speed_kmh = st.slider("Traffic ahead speed (km/h)", float(ms_to_kmh(BASE_RANGES["speed"][0])), float(ms_to_kmh(BASE_RANGES["speed"][1])), key="dt_traffic_speed")
    distance = st.slider("Following distance (m)", float(BASE_RANGES["distance"][0]), float(BASE_RANGES["distance"][1]), key="dt_distance")

    your_speed = kmh_to_ms(your_speed_kmh)
    traffic_speed = kmh_to_ms(traffic_speed_kmh)
    relative_speed = your_speed - traffic_speed
    relative_speed_kmh = your_speed_kmh - traffic_speed_kmh

    st.metric("Relative speed (closing speed)", f"{relative_speed_kmh:.0f} km/h")
    if relative_speed <= 0:
        st.caption("You're travelling at or below the speed of traffic ahead — the gap isn't closing.")
    else:
        st.caption("This is the speed that actually matters for collision risk here — not your raw speedometer reading.")

    render_risk_assessment(clf, reg, "Dynamic Traffic", "Relative speed", relative_speed, distance, weather, road_type, time_of_day, vehicle_type)


def _random_scenario():
    return {
        "weather": random.choice(WEATHER_OPTIONS),
        "road_type": random.choice(ROAD_OPTIONS),
        "time_of_day": random.choice(TIME_OPTIONS),
        "vehicle_type": random.choice(VEHICLE_OPTIONS),
        "speed_kmh": round(random.uniform(ms_to_kmh(BASE_RANGES["speed"][0]) + 10, ms_to_kmh(BASE_RANGES["speed"][1])), 0),
        "distance": round(random.uniform(*BASE_RANGES["distance"]), 0),
    }


def render_quiz(clf, reg):
    if st.button("← Back to home"):
        st.session_state.view = "home"
        st.rerun()

    st.title("🎯 Quiz Mode")
    st.caption("A scenario is shown below. Guess the crash probability before revealing the model's prediction.")

    if "quiz_scenario" not in st.session_state:
        st.session_state.quiz_scenario = _random_scenario()
        st.session_state.quiz_revealed = False
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_rounds = 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total score", st.session_state.quiz_score)
    with c2:
        st.metric("Rounds played", st.session_state.quiz_rounds)

    sc = st.session_state.quiz_scenario
    st.divider()
    st.subheader("Scenario")
    st.write(
        f"- 🌦️ Weather: **{sc['weather']}**\n"
        f"- 🛣️ Road type: **{sc['road_type']}**\n"
        f"- 🌙 Time of day: **{sc['time_of_day']}**\n"
        f"- 🚙 Vehicle: **{sc['vehicle_type']}**\n"
        f"- Speed: **{sc['speed_kmh']:.0f} km/h**\n"
        f"- Following distance: **{sc['distance']:.0f} m**"
    )

    guess = st.slider("Your guess: crash probability (%)", 0, 100, 50, key="quiz_guess", disabled=st.session_state.quiz_revealed)

    if not st.session_state.quiz_revealed:
        if st.button("Reveal prediction", type="primary"):
            st.session_state.quiz_revealed = True
            st.rerun()
    else:
        derived = derived_params_midpoint(sc["weather"], sc["road_type"], sc["time_of_day"], sc["vehicle_type"])
        speed_ms = kmh_to_ms(sc["speed_kmh"])
        actual_prob, _ = predict_risk(
            clf, reg, speed_ms, sc["distance"], derived["reaction_time"],
            derived["brake_eff"], derived["friction"], derived["mass"],
        )
        actual_pct = actual_prob * 100
        diff = abs(guess - actual_pct)
        points = max(0, round(100 - diff))

        st.divider()
        st.metric("Model's actual crash probability", f"{actual_pct:.0f}%")
        st.write(f"Your guess: **{guess}%** — off by **{diff:.0f} points** → **+{points} points**")

        if diff <= 10:
            st.success("🎯 Great intuition!")
        elif diff <= 25:
            st.info("Not bad — getting there.")
        else:
            st.warning("Pretty far off — try exploring the sweep charts in the scenarios to build intuition.")

        if st.button("Next scenario →", type="primary"):
            st.session_state.quiz_score += points
            st.session_state.quiz_rounds += 1
            st.session_state.quiz_scenario = _random_scenario()
            st.session_state.quiz_revealed = False
            st.rerun()


def render_evaluation_dashboard():
    if st.button("← Back to home"):
        st.session_state.view = "home"
        st.rerun()

    st.title("📊 Model Evaluation Dashboard")
    st.caption("How the underlying Stage 1 (classifier) and Stage 2 (regressor) models were trained and compared.")

    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if not os.path.exists(metrics_path):
        st.warning("No metrics_summary.json found — run `python src/models/train.py` locally to generate evaluation artifacts.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    st.subheader("Stage 1 — Crash Classifier")
    st.write(f"**Best model: {metrics['stage1_best_model']}** — chosen for the best ROC-AUC among candidates.")
    st.dataframe(metrics["stage1_comparison"], width='stretch')

    cm_path = os.path.join(RESULTS_DIR, "stage1_confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix — Stage 1", width='content')

    fi1_path = os.path.join(RESULTS_DIR, "stage_1_classifier_feature_importance.png")
    if os.path.exists(fi1_path):
        st.image(fi1_path, caption="Feature Importance — Stage 1", width='content')

    st.divider()
    st.subheader("Stage 2 — Severity Regressor")
    st.write(f"**Best model: {metrics['stage2_best_model']}** — chosen for the best R² among candidates.")
    st.dataframe(metrics["stage2_comparison"], width='stretch')

    fi2_path = os.path.join(RESULTS_DIR, "stage_2_regressor_feature_importance.png")
    if os.path.exists(fi2_path):
        st.image(fi2_path, caption="Feature Importance — Stage 2", width='content')

    st.divider()
    st.caption(f"Features used: {', '.join(metrics['features'])} · Random seed: {metrics['seed']}")


def main():
    if "view" not in st.session_state:
        st.session_state.view = "home"

    if st.session_state.view == "home":
        render_home()
        return

    try:
        clf, reg, used_fallback = get_models()
    except FileNotFoundError:
        st.error(
            "No trained models found. Run `python src/simulation/generate_dataset.py` "
            "then `python src/models/train.py` first to generate data and train the models."
        )
        return

    if used_fallback and not st.session_state.get("_fallback_toast_shown"):
        st.toast("Setting up the model for this session...", icon="⚙️")
        st.session_state["_fallback_toast_shown"] = True

    if st.session_state.view == "emergency":
        render_emergency_braking(clf, reg)
    elif st.session_state.view == "dynamic":
        render_dynamic_traffic(clf, reg)
    elif st.session_state.view == "quiz":
        render_quiz(clf, reg)
    elif st.session_state.view == "evaluation":
        render_evaluation_dashboard()


if __name__ == "__main__":
    main()
