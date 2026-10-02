"""Per-prediction explanations (TreeSHAP) for the sale and rent models.

The models predict log(price) / log(rent), so a SHAP value v is an additive effect on the
log scale: the feature multiplies the prediction by exp(v) relative to the model's baseline
(the average prediction). exp(baseline + sum of all v) = the prediction, exactly.

CatBoost models use CatBoost's own exact TreeSHAP (fast for one row); any other tree model
falls back to the shap package.
"""
import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "state_code": "State", "state": "State",
    "beds": "Bedrooms", "bedrooms": "Bedrooms",
    "baths": "Bathrooms", "bathrooms": "Bathrooms",
    "living_space": "Floor area (sq ft)", "square_feet": "Floor area (sq ft)",
    "zip_code_density": "ZIP population density (per sq mi)",
    "median_household_income": "ZIP household income",
    "latitude": "Latitude", "longitude": "Longitude",
}


def shap_values(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
    """SHAP values (rows x features), the baseline (expected value) and the model inputs."""
    Xt = pipeline[:-1].transform(X)
    model = pipeline[-1]
    if type(model).__name__ == "CatBoostRegressor":
        from catboost import Pool
        pool = Pool(Xt, cat_features=list(model.get_cat_feature_indices()))
        sv = model.get_feature_importance(pool, type="ShapValues")
        return sv[:, :-1], float(sv[0, -1]), Xt
    import shap
    exp = shap.TreeExplainer(model)(Xt)
    return exp.values, float(np.atleast_1d(exp.base_values)[0]), Xt


def top_contributions(pipeline, X: pd.DataFrame, k: int = 5) -> dict:
    """Top-k features by |SHAP| for a single row, in plain units."""
    sv, base, Xt = shap_values(pipeline, X.iloc[[0]])
    row = sv[0]
    order = np.argsort(-np.abs(row))[:k]
    items = []
    for j in order:
        name = Xt.columns[j]
        value = Xt.iloc[0, j]
        items.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": value.item() if hasattr(value, "item") else value,
            "shap_log": float(row[j]),
            "effect_pct": float(np.exp(row[j]) - 1),   # multiplies the prediction by 1 + effect_pct
        })
    return {"baseline": float(np.exp(base)), "prediction": float(np.exp(base + row.sum())),
            "top": items, "other_features_effect_pct": float(np.exp(row[np.setdiff1d(
                np.arange(len(row)), order)].sum()) - 1)}
