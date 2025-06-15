from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pre_process import preprocess_input

model_xg = joblib.load("./models/xgboost_multi_output_model.pkl")
model_rf = joblib.load("./models/randomForest_multi_output_model.pkl")
model_hist = joblib.load("./models/histGBoost_multi_output_model.pkl")

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input from request
        data = request.get_json()

        model_name = data.get("model", "default")
        if model_name == "xgBoost":
            model = model_xg
        elif model_name == "randomForest":
            model = model_rf
        elif model_name == "histGBoost":
            model = model_hist
        else:
            return jsonify({"error": "Unsupported model type"}), 400

        # Preprocess the input
        processed_df = preprocess_input(data)

        # Run model prediction
        prediction_log = model.predict(processed_df)

        # Back-transform to real values
        prediction = np.expm1(prediction_log).astype(int)

        return jsonify(
            {
                "predicted_views": int(prediction[0][0]),
                "predicted_likes": int(prediction[0][1]),
                "predicted_comments": int(prediction[0][2]),
                "predicted_retweets": int(prediction[0][3]),
                "predicted_quotes": int(prediction[0][4]),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
