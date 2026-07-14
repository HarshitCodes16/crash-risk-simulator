"""
Crash Risk Simulator - Streamlit App
--------------------------------------
A what-if scenario simulator (NOT a live telemetry tracker). Home page lets
the user pick between two scenarios:
  - Emergency Braking: the car ahead suddenly stops (original scenario)
  - Dynamic Traffic: both vehicles are moving, risk depends on relative speed
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


@st.cache_resource
def get_models():
    """
    Loads the saved models. If they fail to load (e.g. a scikit-learn/Python
    version mismatch between the machine that trained them and the machine
    running the app), fall back to training fresh, in-memory, using whatever
    library versions are installed here.
    """
    try:
        return load_models()
    except Exception:
        st.info("Saved models weren't compatible with this environment — training fresh models now (a few seconds)...")
        from simulation.generate_dataset import generate_dataset
        from models.train import train_fast_fallback

        df = generate_dataset(n_samples=1500, seed=42)
        clf, reg = train_fast_fallback(df)
        return clf, reg


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


def _param_ranges():
    return {
        "speed": BASE_RANGES["speed"],
        "distance": BASE_RANGES["distance"],
        "reaction_time": (0.3, 5.0),
        "brake_eff": (0.1, 1.0),
        "friction": (0.1, 1.0),
        "mass": (700, 3500),
    }


def render_risk_assessment(clf, reg, speed_label, speed_value, distance, weather, road_type, time_of_day, vehicle_type):
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

    # If closing speed is zero or negative (e.g. moving apart in Dynamic
    # Traffic mode), there is no physical way to collide - skip the model.
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

    speed_result = threshold_crossing(clf, "speed", BASE_RANGES["speed"], fixed_values, threshold=RISK_THRESHOLD)
    safe_speed = speed_result["safe_value"]
    if safe_speed is not None:
        st.info(f"🛡️ Keep {speed_label.lower()} under **{ms_to_kmh(safe_speed):.0f} km/h** to stay below {RISK_THRESHOLD*100:.0f}% risk in these conditions.")
    else:
        st.error(f"🛡️ Even the minimum {speed_label.lower()} is unsafe in these conditions.")

    distance_result = threshold_crossing(clf, "distance", BASE_RANGES["distance"], fixed_values, threshold=RISK_THRESHOLD)
    safe_distance = distance_result["safe_value"]
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

    with st.expander("See sensitivity breakdown for all factors"):
        for param, swing in sorted(swings.items(), key=lambda x: -x[1]):
            label = FEATURE_LABELS[param] if param != "speed" else speed_label
            st.write(f"- **{label}**: {swing*100:.1f} percentage-point swing")


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


def render_emergency_braking(clf, reg):
    if st.button("← Back to home"):
        st.session_state.view = "home"
        st.rerun()

    st.title("🚨 Emergency Braking Simulator")
    st.caption(
        "A what-if scenario tool — not a live tracker. The car ahead suddenly "
        "brakes. Set a hypothetical scenario and see the predicted crash risk."
    )

    st.subheader("Scenario inputs")
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("🌦️ Weather", WEATHER_OPTIONS, index=0, key="eb_weather")
        time_of_day = st.selectbox("🌙 Time of day", TIME_OPTIONS, index=0, key="eb_time")
    with col2:
        road_type = st.selectbox("🛣️ Road type", ROAD_OPTIONS, index=1, key="eb_road")
        vehicle_type = st.selectbox("🚙 Vehicle type", VEHICLE_OPTIONS, index=1, key="eb_vehicle")

    speed_kmh = st.slider("Speed (km/h)", float(ms_to_kmh(BASE_RANGES["speed"][0])), float(ms_to_kmh(BASE_RANGES["speed"][1])), 90.0, key="eb_speed")
    speed = kmh_to_ms(speed_kmh)
    distance = st.slider("Following distance (m)", float(BASE_RANGES["distance"][0]), float(BASE_RANGES["distance"][1]), 20.0, key="eb_distance")

    render_risk_assessment(clf, reg, "Speed", speed, distance, weather, road_type, time_of_day, vehicle_type)


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

    st.subheader("Scenario inputs")
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("🌦️ Weather", WEATHER_OPTIONS, index=0, key="dt_weather")
        time_of_day = st.selectbox("🌙 Time of day", TIME_OPTIONS, index=0, key="dt_time")
    with col2:
        road_type = st.selectbox("🛣️ Road type", ROAD_OPTIONS, index=1, key="dt_road")
        vehicle_type = st.selectbox("🚙 Vehicle type", VEHICLE_OPTIONS, index=1, key="dt_vehicle")

    your_speed_kmh = st.slider("Your speed (km/h)", float(ms_to_kmh(BASE_RANGES["speed"][0])), float(ms_to_kmh(BASE_RANGES["speed"][1])), 108.0, key="dt_your_speed")
    traffic_speed_kmh = st.slider("Traffic ahead speed (km/h)", float(ms_to_kmh(BASE_RANGES["speed"][0])), float(ms_to_kmh(BASE_RANGES["speed"][1])), 100.0, key="dt_traffic_speed")
    distance = st.slider("Following distance (m)", float(BASE_RANGES["distance"][0]), float(BASE_RANGES["distance"][1]), 20.0, key="dt_distance")

    your_speed = kmh_to_ms(your_speed_kmh)
    traffic_speed = kmh_to_ms(traffic_speed_kmh)
    relative_speed = your_speed - traffic_speed
    relative_speed_kmh = your_speed_kmh - traffic_speed_kmh

    st.metric("Relative speed (closing speed)", f"{relative_speed_kmh:.0f} km/h")
    if relative_speed <= 0:
        st.caption("You're travelling at or below the speed of traffic ahead — the gap isn't closing.")
    else:
        st.caption("This is the speed that actually matters for collision risk here — not your raw speedometer reading.")

    render_risk_assessment(clf, reg, "Relative speed", relative_speed, distance, weather, road_type, time_of_day, vehicle_type)


def main():
    if "view" not in st.session_state:
        st.session_state.view = "home"

    if st.session_state.view == "home":
        render_home()
        return

    try:
        clf, reg = get_models()
    except FileNotFoundError:
        st.error(
            "No trained models found. Run `python src/simulation/generate_dataset.py` "
            "then `python src/models/train.py` first to generate data and train the models."
        )
        return

    if st.session_state.view == "emergency":
        render_emergency_braking(clf, reg)
    elif st.session_state.view == "dynamic":
        render_dynamic_traffic(clf, reg)


if __name__ == "__main__":
    main()