import torch
import pandas as pd
import joblib
import json
import os
import numpy as np

from recommendation_personalized_randomforest.deep_learning_model_production import RecommenderNN

def _safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0

def _apply_context_rerank(scores_df, processed_df, weather_val, time_val):
    feature_cols = ["genre_encoded", "language_encoded", "type_encoded"]
    ctx_df = processed_df[
        (processed_df["weather_encoded"] == weather_val) &
        (processed_df["timeOfDay_encoded"] == time_val)
    ]
    if ctx_df.empty:
        ctx_df = processed_df

    top_genres = set(ctx_df["genre_encoded"].value_counts().head(3).index.tolist())
    top_langs = set(ctx_df["language_encoded"].value_counts().head(2).index.tolist())
    preferred_type = int(ctx_df["type_encoded"].mode().iloc[0])

    bonus = np.zeros(len(scores_df), dtype=float)
    bonus += np.where(scores_df["genre_encoded"].isin(top_genres), 0.04, -0.02)
    bonus += np.where(scores_df["language_encoded"].isin(top_langs), 0.06, -0.03)
    bonus += np.where(scores_df["type_encoded"] == preferred_type, 0.02, -0.01)

    out = scores_df.copy()
    out["score"] = (out["score"] + bonus).clip(lower=0.0, upper=1.0)
    return out

def recommend_songs_for_user_enhanced(
    user_id,
    weather,
    timeOfDay,
    top_n=5,
    exclude_listened=True,
    context_rerank=True
):
    USER_FOLDER = os.path.join("recommendation_personalized_randomforest/processed_data", user_id)
    MODEL_PATH = os.path.join("recommendation_personalized_randomforest/model", f"{user_id}/{user_id}_personalized_model_pytorch.pth")
    SONG_MAP_PATH = os.path.join("recommendation_personalized_randomforest/model", f"{user_id}/{user_id}_song_mapping.json")
    HISTORY_PATH = os.path.join("recommendation_personalized_randomforest/listening_history_data", f"{user_id}_listening_data.csv")

    # ===== Load encoders =====
    weather_enc = joblib.load(os.path.join(USER_FOLDER, "weather_encoder.pkl"))
    time_enc = joblib.load(os.path.join(USER_FOLDER, "timeofday_encoder.pkl"))
    genre_enc = joblib.load("recommendation_personalized_randomforest/genre_encoder.pkl")
    language_enc = joblib.load("recommendation_personalized_randomforest/language_encoder.pkl")
    type_enc = joblib.load("recommendation_personalized_randomforest/type_encoder.pkl")

    # ===== Load mapping =====
    # Stored as {class_idx: song_id_encoded}; invert for fast lookup.
    with open(SONG_MAP_PATH, "r") as f:
        idx_to_song_encoded = json.load(f)
    song_encoded_to_idx = {int(v): int(k) for k, v in idx_to_song_encoded.items()}

    num_songs = len(idx_to_song_encoded)

    # ===== Load model =====
    model = RecommenderNN(input_dim=5, hidden_dim=64, output_dim=num_songs)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # ===== Encode weather + time =====
    weather_val = _safe_transform(weather_enc, weather)
    time_val = _safe_transform(time_enc, timeOfDay)
    processed_data = pd.read_csv(os.path.join(USER_FOLDER, f"{user_id}_processed_data.csv"))
    processed_data.columns = processed_data.columns.str.strip()

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

    # Score each candidate by its own song class probability.
    class_indices = all_songs["song_id_encoded"].map(song_encoded_to_idx).fillna(-1).astype(int).to_numpy()
    row_indices = np.arange(len(all_songs))
    valid_mask = class_indices >= 0
    # Fallback for unseen songs (not in user-specific class mapping):
    # use model confidence so scores are still comparable and not forced to 0.
    song_scores = probs.max(axis=1)
    song_scores[valid_mask] = probs[row_indices[valid_mask], class_indices[valid_mask]]
    all_songs["score"] = song_scores

    if context_rerank:
        all_songs = _apply_context_rerank(all_songs, processed_data, weather_val, time_val)

    # Optionally avoid recommending already-listened songs for novelty.
    if exclude_listened and os.path.exists(HISTORY_PATH):
        history_df = pd.read_csv(HISTORY_PATH)
        if "song_id" in history_df.columns:
            listened_song_ids = set(history_df["song_id"].astype(str))
            all_songs = all_songs[~all_songs["song_id"].astype(str).isin(listened_song_ids)]

    # Get top N songs
    top_songs = (
        all_songs
        .drop_duplicates(subset=["song_id"])
        .sort_values("score", ascending=False)
        .head(top_n)
    )

    # Convert to result dict
    result = []
    for _, row in top_songs.iterrows():
        result.append({
            "song_id": row["song_id"],
            "score": float(row["score"])
        })

    return result
