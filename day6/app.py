import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 模型相对路径（与 app.py 同目录）
model = joblib.load("titanic_rf_model.pkl")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title"]
sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 0, "C": 1, "Q": 2}
title_map = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5,
    "Col": 6, "Major": 7, "Mlle": 8, "Countess": 9, "Ms": 10, "Lady": 11
}

app = Flask(__name__)

@app.route("/")
def home():
    return "Titanic Survival Prediction API. Use POST /predict with JSON data."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "Sex" in data and isinstance(data["Sex"], str):
            data["Sex"] = sex_map.get(data["Sex"], 0)
        if "Embarked" in data and isinstance(data["Embarked"], str):
            data["Embarked"] = embarked_map.get(data["Embarked"], 0)
        if "Title" in data and isinstance(data["Title"], str):
            data["Title"] = title_map.get(data["Title"], 12)

        df_input = pd.DataFrame([data])
        for col in features:
            if col not in df_input.columns:
                df_input[col] = 0
        X = df_input[features]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].tolist()
        return jsonify({"prediction": int(pred), "probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)