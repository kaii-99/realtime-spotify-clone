import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("processed_data.csv")

X = df[[
    "user_id_encoded",
    "genre_encoded",
    "language_encoded",
    "type_encoded",
    "weather_encoded",
    "timeOfDay_encoded"
]]

y = df["song_id_encoded"]

#Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(model.feature_names_in_)

joblib.dump(model, "song_recommendation_model.pkl")

# Explanations with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
#
## For multi-class RandomForest: average absolute SHAP values across classes
#shap_values_global = np.mean([np.abs(sv) for sv in shap_values], axis=0)
#
## Plot global feature importance
#shap_values_exp = shap.Explanation(
#    values=shap_values_global,
#    feature_names=X_train.columns,
#    data=X_train.to_numpy()
#)
#
#shap.plots.bar(shap_values_exp)

# Predict probabilities for the user
# user_input must have all 6 columns
user_input = X_train.iloc[0:1]
probs = model.predict_proba(user_input)[0]
top_class_idx = np.argmax(probs)

shap_exp = explainer(user_input)
# shap_exp.values has shape [n_classes, n_features], pick top class
shap_exp_top = shap.Explanation(
    values=shap_exp.values[top_class_idx].reshape(1, -1),  # 1 row, n_features
    base_values=shap_exp.base_values[top_class_idx].reshape(1),
    feature_names=X_train.columns,
    data=user_input
)

shap.plots.bar(shap_exp_top)
