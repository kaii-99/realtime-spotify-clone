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

#USER_ID = "690f6586822864ba799ef3a3"

#INPUT_FOLDER = os.path.join("processed_data", USER_ID)
#MODEL_FOLDER = os.path.join("model", USER_ID)

#os.makedirs(MODEL_FOLDER, exist_ok=True)

#INPUT_FILE   = os.path.join(INPUT_FOLDER, f"{USER_ID}_processed_data.csv")
#MODEL_FILE = os.path.join(MODEL_FOLDER, "song_recommendation_model.pkl")

#all users
USER_IDS = {
    "testuser1": "690f61b5822864ba799ef31d",
    "testuser2": "690f6586822864ba799ef3a3",
    "testuser3": "690f66bf822864ba799ef3fc"
}

BASE_INPUT_FOLDER = "processed_data"
BASE_MODEL_FOLDER = "model"

for username, user_id in USER_IDS.items():
    INPUT_FOLDER = os.path.join(BASE_INPUT_FOLDER, user_id)
    MODEL_FOLDER = os.path.join(BASE_MODEL_FOLDER, user_id)

    INPUT_FILE = os.path.join(INPUT_FOLDER, f"{user_id}_processed_data.csv")
    MODEL_FILE = os.path.join(MODEL_FOLDER, "song_recommendation_model.pkl")

    if not os.path.exists(INPUT_FILE):
        print(f"Processed data not found for {username}")
        continue

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    if len(df) < 10:
        print(f"Not enough data to train model for {username}")
        continue

    X = df[[
        "genre_encoded",
        "language_encoded",
        "type_encoded",
        "weather_encoded",
        "timeOfDay_encoded"
    ]]

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

    print(f"{username} model trained")
    print(f"Accuracy: {acc:.4f}")
    print(f"Features: {model.feature_names_in_}")
    print(f"Importances: {model.feature_importances_}\n")

    joblib.dump(model, MODEL_FILE)