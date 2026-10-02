from flask import Flask, request, jsonify
from flask_cors import CORS
from catboost import CatBoostRegressor, Pool
import os

from preprocess import build_model_input_frame, CATEGORICAL_IN_SCHEMA

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_house_model.cbm")

# Flask app
app = Flask(__name__)
CORS(app)

# Load model
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json(force=True) or {}
    try:
        # Convert JSON payload into model-ready DataFrame
        input_df = build_model_input_frame(request_data)

        # Identify categorical feature indices
        categorical_indices = [input_df.columns.get_loc(col) for col in CATEGORICAL_IN_SCHEMA]

        # Build CatBoost Pool
        data_pool = Pool(input_df, cat_features=categorical_indices if categorical_indices else None)

        # Predict
        prediction_value = float(model.predict(data_pool)[0])

        return jsonify({"prediction_usd": round(prediction_value, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
