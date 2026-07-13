"""
Physics Engine
--------------
Two-car braking scenario (matches the original simulation-based-ml-car-crash logic):
- Front car brakes suddenly.
- Rear car (ours) reacts after `reaction_time`, then brakes with deceleration
  determined by `friction` and `brake_eff`.
- Depending on speed, gap (`distance`), reaction time, and braking, either the
  car stops safely or a collision occurs with some impact force.

This same formula is used TWICE in the project:
1. Inside the simulator, to generate labeled training data.
2. Inside the independent verification layer, to sanity-check the ML model's
   crash-probability output against classical deterministic physics.
"""

G = 9.8  # gravity, m/s^2


def stopping_distance(speed, reaction_time, friction, brake_eff):
    """
    Classical stopping distance formula:
        d = v * t_reaction + v^2 / (2 * mu * eta * g)
    """
    decel = max(friction * brake_eff * G, 0.01)  # avoid divide-by-zero
    reaction_distance = speed * reaction_time
    braking_distance = (speed ** 2) / (2 * decel)
    return reaction_distance + braking_distance


def simulate_scenario(speed, distance, reaction_time, brake_eff, friction, mass):
    """
    Runs the deterministic two-car physics scenario.
    Returns (crash: bool, impact_force: float)
    """
    decel = max(friction * brake_eff * G, 0.01)
    reaction_distance = speed * reaction_time

    if reaction_distance >= distance:
        # Car hasn't even started braking before reaching the gap -> full-speed impact
        v_impact = speed
        crash = True
    else:
        remaining_distance = distance - reaction_distance
        full_stop_distance = (speed ** 2) / (2 * decel)

        if full_stop_distance <= remaining_distance:
            # Car stops safely before reaching the front car
            return False, 0.0
        else:
            # Partial braking happened, but not enough -> impact at reduced speed
            v_squared = speed ** 2 - 2 * decel * remaining_distance
            v_impact = (max(v_squared, 0)) ** 0.5
            crash = True

    if not crash:
        return False, 0.0

    # Impact force approximated via kinetic energy at impact, scaled by an
    # assumed crumple-zone distance (~0.5m) to bring values into a realistic
    # Newton-scale range: F = 0.5 * m * v^2 / crumple_distance
    crumple_distance = 0.5
    impact_force = 0.5 * mass * (v_impact ** 2) / crumple_distance
    return True, impact_force


def verify_against_ml(speed, distance, reaction_time, brake_eff, friction, mass,
                       ml_crash_probability, probability_threshold=0.5,
                       uncertain_band=0.10):
    """
    Independent verification layer.
    Compares the deterministic physics decision against the ML model's
    probability-based decision.

    Rather than a strict binary agree/disagree (which flags "disagreement"
    even for trivial differences when ML probability sits right at the
    threshold), this uses an uncertain zone: if ML probability is within
    `uncertain_band` of the threshold (e.g. 40-60% for a 10% band around a
    50% threshold), the scenario is genuinely borderline - both methods can
    reasonably differ here, and that's reported honestly instead of being
    forced into an agree/disagree verdict.

    Returns a dict with a "status" of one of:
        "agree"      - physics and ML both land on the same side, clearly
        "uncertain"  - ML probability is in the borderline band near the threshold
        "disagree"   - ML is confidently on one side, physics says the other
    """
    physics_crash, physics_force = simulate_scenario(
        speed, distance, reaction_time, brake_eff, friction, mass
    )
    ml_crash_decision = ml_crash_probability >= probability_threshold

    lower = probability_threshold - uncertain_band
    upper = probability_threshold + uncertain_band
    is_borderline = lower <= ml_crash_probability <= upper

    if is_borderline:
        status = "uncertain"
    elif physics_crash == ml_crash_decision:
        status = "agree"
    else:
        status = "disagree"

    return {
        "status": status,
        "physics_crash": physics_crash,
        "physics_impact_force": physics_force,
        "ml_crash_decision": ml_crash_decision,
        "agrees": physics_crash == ml_crash_decision,
        "stopping_distance": stopping_distance(speed, reaction_time, friction, brake_eff),
    }
