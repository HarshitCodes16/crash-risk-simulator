"""
Prediction Layer
-----------------
Loads the trained Stage 1 / Stage 2 models and provides higher-level
functions used directly by the Streamlit app:
- predict_risk(): crash probability + severity for a given scenario
- sensitivity_sweep(): sweep one parameter across its range, holding others fixed
- biggest_risk_factor(): which of the 6 inputs is swinging risk the most right now
- safe_speed_recommendation(): reverse-lookup max speed under a risk threshold
"""

import os
import joblib
import numpy as np
import pandas as pd

FEATURES = ["speed", "distance", "reaction_time", "brake_eff", "friction", "mass"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def load_models():
    clf = joblib.load(os.path.join(MODELS_DIR, "stage1_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "stage2_regressor.pkl"))
    return clf, reg


def _to_row(speed, distance, reaction_time, brake_eff, friction, mass):
    values = [speed, distance, reaction_time, brake_eff, friction, mass]
    return pd.DataFrame([values], columns=FEATURES)


def predict_risk(clf, reg, speed, distance, reaction_time, brake_eff, friction, mass):
    row = _to_row(speed, distance, reaction_time, brake_eff, friction, mass)
    crash_probability = clf.predict_proba(row)[0, 1]
    severity = reg.predict(row)[0] if crash_probability >= 0.5 else 0.0
    return crash_probability, severity


def sensitivity_sweep(clf, param_name, param_range, fixed_values, n_points=25):
    """
    Sweep `param_name` across param_range (min, max), holding all other
    params at fixed_values (a dict of the 6 feature values). Returns
    (x_values, probabilities).
    """
    x_values = np.linspace(param_range[0], param_range[1], n_points)
    probs = []
    for x in x_values:
        values = dict(fixed_values)
        values[param_name] = x
        row = _to_row(*[values[f] for f in FEATURES])
        probs.append(clf.predict_proba(row)[0, 1])
    return x_values, np.array(probs)


def biggest_risk_factor(clf, fixed_values, param_ranges):
    """
    For the current scenario, sweep each of the 6 parameters individually
    across its full range (others held fixed) and measure how much crash
    probability swings. Returns the name of the parameter with the largest
    swing, plus a dict of all swings for transparency.
    """
    swings = {}
    for param in FEATURES:
        _, probs = sensitivity_sweep(clf, param, param_ranges[param], fixed_values, n_points=15)
        swings[param] = probs.max() - probs.min()
    biggest = max(swings, key=swings.get)
    return biggest, swings


def threshold_crossing(clf, param_name, param_range, fixed_values, threshold=0.5, n_points=100):
    """
    Generalized safe-value finder for ANY of the 6 parameters (not just speed).

    Sweeps `param_name` across its range and finds where crash probability
    crosses `threshold`. Works regardless of whether increasing the parameter
    increases risk (e.g. speed, mass) or decreases risk (e.g. distance,
    friction, brake_eff) - it detects the direction automatically.

    Returns a dict: {
        "safe_value": the boundary value recommended to stay under threshold
                       (None if no safe value exists in range),
        "direction": "higher_is_riskier" or "lower_is_riskier",
        "x_values": swept x values,
        "probabilities": corresponding crash probabilities,
    }
    """
    x_values, probs = sensitivity_sweep(clf, param_name, param_range, fixed_values, n_points=n_points)

    # Detect direction: does risk generally increase or decrease as param increases?
    higher_is_riskier = probs[-1] >= probs[0]

    safe_mask = probs < threshold
    if not safe_mask.any():
        safe_value = None
    elif higher_is_riskier:
        # want the largest x that's still safe
        safe_value = float(x_values[safe_mask].max())
    else:
        # want the smallest x that's still safe
        safe_value = float(x_values[safe_mask].min())

    return {
        "safe_value": safe_value,
        "direction": "higher_is_riskier" if higher_is_riskier else "lower_is_riskier",
        "x_values": x_values,
        "probabilities": probs,
    }


def safe_speed_recommendation(clf, fixed_values, speed_range, threshold=0.5, n_points=60):
    """
    Reverse-lookup: sweep speed across its range (holding other params fixed)
    and find the highest speed at which crash probability stays below threshold.
    Returns None if even the minimum speed exceeds the threshold.
    """
    result = threshold_crossing(clf, "speed", speed_range, fixed_values, threshold, n_points)
    return result["safe_value"]
