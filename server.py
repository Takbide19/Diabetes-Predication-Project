from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

# Load the dataset
csv_file = "src\\Healthcare-Diabetes.csv"
df = pd.read_csv(csv_file)
X = df.drop(columns=["Outcome"]).values
y = df["Outcome"].values

# Preprocessing: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.3, random_state=42,
)

# Feature Selection (Top 5 Features)
select_k_best = SelectKBest(score_func=f_classif, k=5)
X_train_selected = select_k_best.fit_transform(X_train, y_train)
X_test_selected = select_k_best.transform(X_test)
selected_features = (
    df.drop(columns=["Outcome"]).columns[select_k_best.get_support()].tolist()
)

# Train Random Forest Model on Selected Features
rf_feature_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feature_importance.fit(X_train_selected, y_train)
feature_importances = rf_feature_importance.feature_importances_

# Feature Extraction (PCA to 3 Components)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_selected)
X_test_pca = pca.transform(X_test_selected)

# Train Random Forest Model on PCA Components
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

# Save the trained model using Pickle
with open("diabetes_rf_model.pkl", "wb") as model_file:
    pickle.dump(rf, model_file)

# Evaluate Model
cv_scores = cross_val_score(rf, X_train_pca, y_train, cv=5)
y_pred = rf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Cross-Validation Scores:", cv_scores)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Modify input based on gender
        if data["gender"] == "male":
            input_data = np.array([
                [
                    0,  # Set pregnancies to 0 for male
                    data["glucose"],
                    data["bloodPressure"],
                    data["skinThickness"],
                    data["insulin"],
                    data["bmi"],
                    data["diabetesPedigreeFunction"],
                    data["age"],
                ]
            ])
        else:
            input_data = np.array([
                [
                    data["pregnancies"],
                    data["glucose"],
                    data["bloodPressure"],
                    data["skinThickness"],
                    data["insulin"],
                    data["bmi"],
                    data["diabetesPedigreeFunction"],
                    data["age"],
                ]
            ])

        # Apply preprocessing (scaling, feature selection, PCA)
        input_scaled = scaler.transform(input_data)
        input_selected = select_k_best.transform(input_scaled)
        input_pca = pca.transform(input_selected)

        # Load the saved model using Pickle
        with open("diabetes_rf_model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)

        # Prediction
        prediction = rf.predict(input_pca)
        result = (
            "Oops! You are at risk for diabetes."
            if prediction[0] == 1
            else "No diabetes detected!"
        )

        return jsonify({"result": result, "selected_features": selected_features})

    except Exception as e:
        error_message = f"Error processing the input: {str(e)}"
        print(error_message)
        return (
            jsonify({"result": "Error processing the input.", "error": str(e)}),
            400,
        )

@app.route("/model-accuracy", methods=["GET"])
def model_accuracy():
    try:
        return jsonify({
            "accuracy": accuracy * 100,
            "cross_validation_scores": cv_scores.tolist(),
            "classification_report": classification_report_str
        })
    except Exception as e:
        error_message = f"Error generating accuracy data: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 400

@app.route("/chart-data", methods=["GET"])
def chart_data():
    try:
        # Feature importance from feature selection phase
        feature_importance_data = {
            "labels": selected_features,
            "values": feature_importances.tolist()
        }

        # Outcome distribution
        outcome_counts = df["Outcome"].value_counts().to_dict()

        return jsonify({
            "feature_importance": feature_importance_data,
            "outcome_distribution": {
                "labels": ["No Diabetes", "Diabetes"],
                "values": [outcome_counts.get(0, 0), outcome_counts.get(1, 0)]
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error generating chart data: {str(e)}"}), 500
    
@app.route("/correlation-heatmap", methods=["GET"])
def correlation_heatmap():
    try:
        # Compute the correlation matrix
        correlation_matrix = df.corr()
        correlation_dict = correlation_matrix.to_dict()
        return jsonify({"correlation_matrix": correlation_dict})
    except Exception as e:
        error_message = f"Error generating correlation heatmap: {str(e)}"
        return jsonify({"error": error_message}), 500


@app.route("/")
def home():
    return "API is running. Use /chart-data to fetch chart data."

if __name__ == "__main__":
    app.run(debug=True)
