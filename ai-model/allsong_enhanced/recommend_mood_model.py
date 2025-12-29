import numpy as np
import pandas as pd
import joblib

model = joblib.load("allsong_enhanced/song_recommendation_model.pkl")
all_songs = pd.read_csv("allsong_enhanced/all_songs_encoded.csv")
user_enc = joblib.load("allsong_enhanced/user_encoder.pkl")
weather_enc = joblib.load("allsong_enhanced/weather_encoder.pkl")
time_enc = joblib.load("allsong_enhanced/timeofday_encoder.pkl")

def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Fallback to most common / first class
        return 0

def encode_user_inputs(user_id, weather, time_of_day):
    user_id_encoded = safe_transform(user_enc, user_id)
    weather_encoded = safe_transform(weather_enc, weather)
    time_encoded = safe_transform(time_enc, time_of_day)
    return user_id_encoded, weather_encoded, time_encoded

def recommend_songs_for_user(user_id, weather, timeOfDay, top_n=5):
    inputs = []

    user_id_encoded, weather_encoded, timeOfDay_encoded = encode_user_inputs(user_id, weather, timeOfDay)

    for _, song in all_songs.iterrows():
        inputs.append([
            user_id_encoded,
            song["genre_encoded"],
            song["language_encoded"],
            song["type_encoded"],
            weather_encoded,
            timeOfDay_encoded
        ])

    feature_names = [
    "user_id_encoded",
    "genre_encoded",
    "language_encoded",
    "type_encoded",
    "weather_encoded",
    "timeOfDay_encoded"
    ]
    
    inputs_df = pd.DataFrame(inputs, columns=feature_names)
    probs = model.predict_proba(inputs_df)
    song_scores = probs.max(axis=1)

    all_songs["score"] = song_scores
    results = all_songs.sort_values("score", ascending=False).head(top_n)

    return results[["song_id", "score"]].to_dict(orient="records")