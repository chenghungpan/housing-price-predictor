from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("model/model.pkl")

# Load the list of features from model_features.json
with open("model/model_features.json", "r") as feature_file:
    features = json.load(feature_file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        # print('received data:', data)

        # # Check if all required features are present in the input data
        # missing_features = [feature for feature in features if feature not in data]
        # if missing_features:
        #     return jsonify({"error": f"Missing features: {', '.join(missing_features)}"})

        # Create a DataFrame with the input data
        # input_data = {feature: [data[feature]] for feature in features}

        input_df = pd.DataFrame(data)

        # Make predictions using the pre-trained model
        predictions = model.predict(input_df)

        # Prepare the response JSON with both prediction and metadata
        response = {
            "prediction": predictions.tolist(),
            "metadata": {
                "model": "xgboost",
                "app_type": "green_type"
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

