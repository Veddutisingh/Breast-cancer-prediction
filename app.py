from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


app = Flask(__name__)

DATA_PATH = Path(__file__).resolve().parent / "breast_cancer.csv"
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=["id", "Unnamed: 32"], errors="ignore")
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

feature_names = [column for column in data.columns if column != "diagnosis"]
X = data[feature_names]
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Fit the scaler from training rows only.
feature_means = X_train.mean(axis=0).to_numpy()
feature_stds = X_train.std(axis=0, ddof=0).to_numpy(copy=True)
feature_stds[feature_stds == 0] = 1
X_train_scaled = (X_train.to_numpy() - feature_means) / feature_stds
X_test_scaled = (X_test.to_numpy() - feature_means) / feature_stds

logistic_model = LogisticRegression(max_iter=1000)
svm_model = SVC(probability=True)
logistic_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)

logistic_test_predictions = logistic_model.predict(X_test_scaled)
svm_test_predictions = svm_model.predict(X_test_scaled)


def model_metrics(predictions):
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


stats = {
    "logistic_regression": model_metrics(logistic_test_predictions),
    "svm": model_metrics(svm_test_predictions),
}


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/features")
def get_features():
    return jsonify({"features": feature_names})


@app.get("/sample")
def get_sample():
    sample_type = request.args.get("type", "").lower()
    class_value = {"benign": 0, "malignant": 1}.get(sample_type)
    if class_value is None:
        return jsonify({"error": "Choose a benign or malignant sample type."}), 400

    matching_indexes = y_test[y_test == class_value].index.to_numpy()
    sample_index = np.random.default_rng().choice(matching_indexes)
    sample_values = X_test.loc[sample_index, feature_names].astype(float).tolist()

    return jsonify(
        {
            "type": sample_type,
            "features": feature_names,
            "values": sample_values,
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "values" not in payload:
        return jsonify({"error": "Send a JSON object with a 'values' list."}), 400

    values = payload["values"]
    if not isinstance(values, list):
        return jsonify({"error": "'values' must be a list of numbers."}), 400
    if len(values) != len(feature_names):
        return jsonify(
            {"error": f"Expected {len(feature_names)} values, but received {len(values)}."}
        ), 400
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        return jsonify({"error": "Every feature value must be a number."}), 400
    if not np.isfinite(values).all():
        return jsonify({"error": "Feature values must be finite numbers."}), 400

    scaled_values = (np.asarray(values, dtype=float) - feature_means) / feature_stds
    model_input = scaled_values.reshape(1, -1)

    logistic_prediction = int(logistic_model.predict(model_input)[0])
    svm_prediction = int(svm_model.predict(model_input)[0])
    logistic_probability = float(logistic_model.predict_proba(model_input)[0][1])
    svm_probability = float(svm_model.predict_proba(model_input)[0][1])

    return jsonify(
        {
            "logistic_regression": {
                "prediction": "Malignant" if logistic_prediction else "Benign",
                "probability": logistic_probability,
            },
            "svm": {
                "prediction": "Malignant" if svm_prediction else "Benign",
                "probability": svm_probability,
            },
        }
    )


@app.get("/stats")
def get_stats():
    return jsonify(stats)


if __name__ == "__main__":
    app.run()
