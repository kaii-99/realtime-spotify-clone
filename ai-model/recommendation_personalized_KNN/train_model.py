import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

USER_ID = "690f6586822864ba799ef3a3"

INPUT_FOLDER = os.path.join("processed_data", USER_ID)
MODEL_FOLDER = os.path.join("model", USER_ID)

os.makedirs(MODEL_FOLDER, exist_ok=True)

INPUT_FILE   = os.path.join(INPUT_FOLDER, f"{USER_ID}_processed_data.csv")
MODEL_FILE = os.path.join(MODEL_FOLDER, "song_recommendation_model.pkl")

# Load processed data
df = pd.read_csv(INPUT_FILE)

X = df[[
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

model = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(model.feature_names_in_)

joblib.dump(model, MODEL_FILE)