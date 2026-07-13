"""
Data Generation via Simulation
-------------------------------
Generates a labeled dataset of two-car braking scenarios using SimPy for the
discrete-event timing (reaction delay -> braking event), and the shared
physics engine to compute the actual outcome (crash / impact force).

Each row also carries the 4 real-world categorical factors it was generated
under (weather, road_type, time_of_day, vehicle_type) for reference/analysis,
even though the ML models are trained only on the 6 numeric physics features.
"""

import random
import simpy
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.dependency_map import (
    BASE_RANGES, WEATHER_OPTIONS, ROAD_OPTIONS, TIME_OPTIONS, VEHICLE_OPTIONS,
    sample_derived_params,
)
from physics.crash_physics import simulate_scenario


def _run_single_scenario(env, rng, results):
    """A single SimPy process: models the reaction-delay timing, then hands
    off to the physics engine to resolve the outcome."""
    weather = rng.choice(WEATHER_OPTIONS)
    road_type = rng.choice(ROAD_OPTIONS)
    time_of_day = rng.choice(TIME_OPTIONS)
    vehicle_type = rng.choice(VEHICLE_OPTIONS)

    speed = rng.uniform(*BASE_RANGES["speed"])
    distance = rng.uniform(*BASE_RANGES["distance"])

    derived = sample_derived_params(weather, road_type, time_of_day, vehicle_type, rng=rng)
    reaction_time = derived["reaction_time"]
    friction = derived["friction"]
    mass = derived["mass"]
    brake_eff = derived["brake_eff"]

    # Model the reaction delay as a simulated timeout (discrete-event timing)
    yield env.timeout(reaction_time)

    crash, impact_force = simulate_scenario(
        speed, distance, reaction_time, brake_eff, friction, mass
    )

    results.append({
        "weather": weather,
        "road_type": road_type,
        "time_of_day": time_of_day,
        "vehicle_type": vehicle_type,
        "speed": speed,
        "distance": distance,
        "reaction_time": reaction_time,
        "brake_eff": brake_eff,
        "friction": friction,
        "mass": mass,
        "crash": int(crash),
        "impact_force": impact_force,
    })


def generate_dataset(n_samples=4000, seed=42):
    rng = random.Random(seed)
    env = simpy.Environment()
    results = []

    for _ in range(n_samples):
        env.process(_run_single_scenario(env, rng, results))

    env.run()

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=4000, seed=42)
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "crash_simulation_data_v2.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Crash rate: {df['crash'].mean():.2%}")
