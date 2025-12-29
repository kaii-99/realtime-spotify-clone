import numpy as np
import pandas as pd
import joblib
import os
import requests

all_songs = pd.read_csv("group_playlist_randomforest/all_songs_encoded.csv")

def fetch_group_members(group_id):
    url = f"http://localhost:5000/api/export/group_member/{group_id}"
    res = requests.get(url)

    if res.status_code != 200:
        raise Exception("Failed to fetch group members")

    data = res.json()
    return data["members"]

def aggregate_group_scores(user_score_dfs, alpha=0.7):
    merged = user_score_dfs[0].copy()
    merged = merged.rename(columns={"score": "score_0"})

    for i, df in enumerate(user_score_dfs[1:], start=1):
        merged = merged.merge(
            df.rename(columns={"score": f"score_{i}"}),
            on="song_id",
            how="inner"
        )

    score_cols = [c for c in merged.columns if c.startswith("score_")]

    merged["avg_score"] = merged[score_cols].mean(axis=1)
    merged["min_score"] = merged[score_cols].min(axis=1)

    merged["final_score"] = alpha * merged["avg_score"] + (1 - alpha) * merged["min_score"]

    return merged.sort_values("final_score", ascending=False)

def recommend_songs_for_group_ml(group_id, top_n=10, alpha=0.7):

    members = fetch_group_members(group_id)

    all_user_scores = []

    for user_id in members:
        model_path = os.path.join("group_playlist_randomforest/model", user_id, "song_recommendation_model.pkl")
        if not os.path.exists(model_path):
            print(f"No ML model for user {user_id}")
            continue

        model = joblib.load(model_path)

        # Use same song pool for everyone
        candidate_songs = all_songs.copy().reset_index(drop=True)
        inputs_df = candidate_songs[
            ["genre_encoded", "language_encoded", "type_encoded"]
        ]

        probs = model.predict_proba(inputs_df)
        scores = probs.max(axis=1)  # confidence score

        user_scores = candidate_songs[["song_id"]].copy()
        user_scores["score"] = scores

        all_user_scores.append(user_scores)

    if len(all_user_scores) < 2:
        print("⚠️ Not enough users with models")
        return []

    group_scores = aggregate_group_scores(all_user_scores, alpha)

    return group_scores.head(top_n)[
        ["song_id", "final_score", "avg_score", "min_score"]
    ]