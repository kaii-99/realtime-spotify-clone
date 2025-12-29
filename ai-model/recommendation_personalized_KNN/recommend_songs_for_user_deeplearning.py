import torch
import pandas as pd
import joblib
import json
import os

from recommendation_personalized_randomforest.deep_learning_model_production import RecommenderNN

def recommend_songs_for_user_enhanced(user_id, weather, timeOfDay, top_n=5):
    USER_FOLDER = os.path.join("recommendation_personalized_randomforest/processed_data", user_id)
    MODEL_PATH = os.path.join("recommendation_personalized_randomforest/model", f"{user_id}/{user_id}_personalized_model.pth")
    SONG_MAP_PATH = os.path.join("recommendation_personalized_randomforest/model", f"{user_id}/{user_id}_song_mapping.json")

    # ===== Load encoders =====
    weather_enc = joblib.load(os.path.join(USER_FOLDER, "weather_encoder.pkl"))
    time_enc = joblib.load(os.path.join(USER_FOLDER, "timeofday_encoder.pkl"))
    genre_enc = joblib.load("recommendation_personalized_randomforest/genre_encoder.pkl")
    language_enc = joblib.load("recommendation_personalized_randomforest/language_encoder.pkl")
    type_enc = joblib.load("recommendation_personalized_randomforest/type_encoder.pkl")

    # ===== Load mapping =====
    with open(SONG_MAP_PATH, "r") as f:
        song_id_to_idx = json.load(f)
    idx_to_song_id = {int(v): str(k) for k, v in song_id_to_idx.items()}

    num_songs = len(song_id_to_idx)

    # ===== Load model =====
    model = RecommenderNN(input_dim=5, hidden_dim=64, output_dim=num_songs)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

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
    
    X = torch.tensor(inputs, dtype=torch.float32)

    # ===== Predict =====
    with torch.no_grad():
        scores = model(X)
        probs = torch.softmax(scores, dim=1).numpy()

    # Take max probability for each song (axis=1)
    song_scores = probs.max(axis=1)
    all_songs["score"] = song_scores

    # Get top N songs
    top_songs = all_songs.sort_values("score", ascending=False).head(top_n)

    # Convert to result dict
    result = []
    for _, row in top_songs.iterrows():
        result.append({
            "song_id": row["song_id"],
            "score": float(row["score"])
        })

    return result
