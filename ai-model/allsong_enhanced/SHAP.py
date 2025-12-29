import pandas as pd
import joblib
import shap
import numpy as np

# Load model and encoders
model = joblib.load("song_recommendation_model.pkl")

# Example user input (shape [1, n_features])
user_input = np.array([[0, 4, 0, 0, 0, 1]])

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values for the input
shap_values = explainer(user_input)  # ✅ new SHAP style

# Predict probabilities
probs = model.predict_proba(user_input)[0]

# Pick top N songs
top_n = 5
top_indices = np.argsort(probs)[-top_n:]  # positions in model.classes_

# Loop through top predictions and plot SHAP
for class_idx in top_indices:
    # Extract values for this class and this instance
    values = shap_values.values[0, :, class_idx]   # 1 row, all features for this class
    base_value = shap_values.base_values[0, class_idx]
    
    exp = shap.Explanation(
        values=values,
        base_values=base_value,
        data=user_input[0],
        feature_names=['user_id', 'genre', 'language', 'type', 'weather', 'timeOfDay']
    )
    
    print(f"Top song class {model.classes_[class_idx]} SHAP values:")
    shap.plots.bar(exp)
