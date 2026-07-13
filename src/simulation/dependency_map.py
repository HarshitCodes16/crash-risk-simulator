"""
Dependency Mapping Layer
------------------------
Converts the 4 real-world categorical factors (weather, road type, time of day,
vehicle type) into the 6 raw physics parameters that the original simulation
uses (speed, distance, reaction_time, brake_eff, friction, mass).

Design decisions (from project blueprint):
- Multiplicative stacking: effects compound (base * factor1 * factor2 ...)
- Range-based sampling: each condition maps to a numeric RANGE, not a fixed value
- Hard clamps: every derived parameter is clamped to a physically valid range
- Weather affects friction AND reaction_time (visibility)
- Road type affects friction
- Time of day affects reaction_time (independent of weather) -> stacks with it
- Vehicle type affects mass AND brake_eff
"""

import random

# ---- Base physical ranges (from the original simulation) ----
BASE_RANGES = {
    "speed": (10, 40),          # m/s
    "distance": (5, 220),       # m (widened from the original 5-50 so that, combined with
                                 # the 10-40 m/s speed range, the dataset produces a realistic
                                 # mix of safe-stop and crash outcomes instead of near-always-crash)
    "reaction_time": (0.5, 2.5),  # s
    "brake_eff": (0.5, 1.0),     # unitless
    "friction": (0.3, 0.9),      # unitless
    "mass": (800, 2000),         # kg
}

# ---- Hard clamps (physically valid bounds, prevents multiplicative stacking blowup) ----
CLAMPS = {
    "reaction_time": (0.3, 5.0),
    "brake_eff": (0.1, 1.0),
    "friction": (0.1, 1.0),
    "mass": (700, 3500),
}

# ---- Categorical multiplier tables ----
WEATHER_FACTORS = {
    # weather -> (friction_multiplier, reaction_time_multiplier)
    "Sunny":  (1.00, 1.00),
    "Rainy":  (0.65, 1.25),
    "Foggy":  (0.85, 1.40),
    "Icy":    (0.45, 1.15),
}

ROAD_FACTORS = {
    # road type -> friction_multiplier
    "Highway": 1.10,
    "City":    0.95,
    "Hilly":   0.80,
}

TIME_FACTORS = {
    # time of day -> reaction_time_multiplier (independent of weather)
    "Day":   1.00,
    "Night": 1.30,
}

VEHICLE_FACTORS = {
    # vehicle type -> (mass_range, brake_eff_multiplier)
    "Hatchback": ((800, 1100), 1.05),
    "Sedan":     ((1100, 1500), 1.00),
    "SUV":       ((1500, 2000), 0.90),
    "Truck":     ((2000, 3200), 0.75),
}


def _clamp(value, key):
    if key in CLAMPS:
        lo, hi = CLAMPS[key]
        return max(lo, min(hi, value))
    return value


def sample_derived_params(weather, road_type, time_of_day, vehicle_type, rng=None):
    """
    Given the 4 categorical choices, sample the 4 derived physics parameters:
    friction, reaction_time, mass, brake_eff.
    (speed and distance are NOT derived here - they're direct user/sim inputs.)

    Returns a dict: {friction, reaction_time, mass, brake_eff}
    """
    r = rng if rng is not None else random

    # --- friction: base range * weather * road ---
    base_friction = r.uniform(*BASE_RANGES["friction"])
    w_fric, w_react = WEATHER_FACTORS[weather]
    r_fric = ROAD_FACTORS[road_type]
    friction = _clamp(base_friction * w_fric * r_fric, "friction")

    # --- reaction_time: base range * weather * time_of_day ---
    base_reaction = r.uniform(*BASE_RANGES["reaction_time"])
    t_react = TIME_FACTORS[time_of_day]
    reaction_time = _clamp(base_reaction * w_react * t_react, "reaction_time")

    # --- mass: vehicle-specific range ---
    mass_range, veh_brake_mult = VEHICLE_FACTORS[vehicle_type]
    mass = _clamp(r.uniform(*mass_range), "mass")

    # --- brake_eff: base range * vehicle multiplier ---
    base_brake = r.uniform(*BASE_RANGES["brake_eff"])
    brake_eff = _clamp(base_brake * veh_brake_mult, "brake_eff")

    return {
        "friction": friction,
        "reaction_time": reaction_time,
        "mass": mass,
        "brake_eff": brake_eff,
    }


def derived_params_midpoint(weather, road_type, time_of_day, vehicle_type):
    """
    Deterministic version used by the live app: instead of randomly sampling,
    use the midpoint of each range so the same dropdown selection always gives
    the same result (no flicker between reruns for the same inputs).
    """
    w_fric, w_react = WEATHER_FACTORS[weather]
    r_fric = ROAD_FACTORS[road_type]
    mid_friction = sum(BASE_RANGES["friction"]) / 2
    friction = _clamp(mid_friction * w_fric * r_fric, "friction")

    t_react = TIME_FACTORS[time_of_day]
    mid_reaction = sum(BASE_RANGES["reaction_time"]) / 2
    reaction_time = _clamp(mid_reaction * w_react * t_react, "reaction_time")

    mass_range, veh_brake_mult = VEHICLE_FACTORS[vehicle_type]
    mass = _clamp(sum(mass_range) / 2, "mass")

    mid_brake = sum(BASE_RANGES["brake_eff"]) / 2
    brake_eff = _clamp(mid_brake * veh_brake_mult, "brake_eff")

    return {
        "friction": friction,
        "reaction_time": reaction_time,
        "mass": mass,
        "brake_eff": brake_eff,
    }


WEATHER_OPTIONS = list(WEATHER_FACTORS.keys())
ROAD_OPTIONS = list(ROAD_FACTORS.keys())
TIME_OPTIONS = list(TIME_FACTORS.keys())
VEHICLE_OPTIONS = list(VEHICLE_FACTORS.keys())
