import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

PROCESSED_ROOT = "processed_data"
MODEL_ROOT = "model"

os.makedirs(MODEL_ROOT, exist_ok=True)

# Loop through all users
for USER_ID in os.listdir(PROCESSED_ROOT):
    USER_FOLDER = os.path.join(PROCESSED_ROOT, USER_ID)

    if not os.path.isdir(USER_FOLDER):
        continue

    INPUT_FILE = os.path.join(USER_FOLDER, f"{USER_ID}_processed_data.csv")
    MODEL_FOLDER = os.path.join(MODEL_ROOT, USER_ID)
    MODEL_FILE = os.path.join(MODEL_FOLDER, "song_recommendation_model.pkl")

    if not os.path.exists(INPUT_FILE):
        print(f"Missing data for {USER_ID}")
        continue

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    print(f"\nTraining model for user: {USER_ID}")

    df = pd.read_csv(INPUT_FILE)

    if len(df) < 10:
        print("Not enough data, skipped")
        continue

    # Features (NO mood)
    X = df[
        [
            "genre_encoded",
            "language_encoded",
            "type_encoded",
            "weather_encoded",
            "timeOfDay_encoded"
        ]
    ]

    y = df["song_id_encoded"]

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
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print("Features:", model.feature_names_in_)

    joblib.dump(model, MODEL_FILE)

    print(f"Model saved → {MODEL_FILE}")

print("\nAll user models trained")