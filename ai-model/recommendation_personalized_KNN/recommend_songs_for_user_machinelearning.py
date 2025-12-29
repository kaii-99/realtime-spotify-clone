import numpy as np
import pandas as pd
import joblib
import os

def recommend_songs_for_user_ml(user_id, weather, timeOfDay, top_n=5):

    USER_FOLDER = os.path.join("recommendation_personalized_randomforest/processed_data", user_id)
    MODEL_PATH = os.path.join("recommendation_personalized_randomforest/model", f"{user_id}/song_personalized_model.pkl")

    # ===== Check model exists =====
    if not os.path.exists(MODEL_PATH):
        raise Exception(f"No personalized ML model found for user {user_id}")

    # ===== Load encoders =====
    weather_enc = joblib.load(os.path.join(USER_FOLDER, "weather_encoder.pkl"))
    time_enc = joblib.load(os.path.join(USER_FOLDER, "timeofday_encoder.pkl"))
    genre_enc = joblib.load("recommendation_personalized_randomforest/genre_encoder.pkl")
    language_enc = joblib.load("recommendation_personalized_randomforest/language_encoder.pkl")
    type_enc = joblib.load("recommendation_personalized_randomforest/type_encoder.pkl")

    # ===== Load model =====
    model = joblib.load(MODEL_PATH)

    # ===== Encode weather + time =====
    weather_val = weather_enc.transform([weather])[0]
    time_val = time_enc.transform([timeOfDay])[0]

    # ===== Load all songs features =====
    all_songs = pd.read_csv("recommendation_personalized_randomforest/all_songs_encoded.csv")
    all_songs.columns = all_songs.columns.str.strip()

    inputs = []
    for _, song in all_songs.iterrows():
        inputs.append([
            song["genre_encoded"],
            song["language_encoded"],
            song["type_encoded"],
            weather_val,
            time_val
        ])
    
    feature_names = [
        "genre_encoded",
        "language_encoded",
        "type_encoded",
        "weather_encoded",
        "timeOfDay_encoded"
    ]

    inputs_df = pd.DataFrame(inputs, columns=feature_names)

    # ===== Predict =====
    probs = model.predict_proba(inputs_df)

    # probability of positive class (like)
    song_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    all_songs["score"] = song_scores

    # Get top N songs
    top_songs = all_songs.sort_values("score", ascending=False).head(top_n)

    return top_songs[["song_id", "score"]].to_dict(orient="records")