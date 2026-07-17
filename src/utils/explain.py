"""
Per-Scenario Explainability (SHAP)
------------------------------------
Unlike the global "risk contribution" bars (which show how much each factor
swings crash probability across its FULL range), this explains ONE specific
prediction: for the exact scenario the user has set up right now, how much
did each feature push the probability up or down from the model's baseline?

A single, model-agnostic approach is used for all 4 possible Stage 1 model
types (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting -
whichever wins the comparison in a given training run): SHAP's unified
Explainer wraps the model's own predict_proba output directly, so results
are always in probability-space and always reconstruct to the actual
predicted probability, regardless of which model type is active.

This is intentionally NOT auto-computed on every rerun - for Logistic
Regression specifically, this can take several seconds (SHAP has no fast
closed-form path for a black-box-wrapped linear model), so it's triggered
on demand via a button in the UI rather than blocking every scenario change.

Every call is wrapped so that ANY failure (missing shap package, an
unexpected model type, a version incompatibility) returns None instead of
raising - this feature must never be able to crash the app.
"""

FEATURES = ["speed", "distance", "reaction_time", "brake_eff", "friction", "mass"]


def explain_prediction(model, background_df, instance_df):
    """
    Returns {"base": float, "values": {feature: contribution_in_probability_points}}
    or None if an explanation isn't available right now.
    """
    try:
        import shap
        proba_fn = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.Explainer(proba_fn, background_df)
        sv = explainer(instance_df)
        base = float(sv.base_values[0])
        values = {f: float(v) for f, v in zip(FEATURES, sv.values[0])}
        return {"base": base, "values": values}
    except Exception:
        return None
