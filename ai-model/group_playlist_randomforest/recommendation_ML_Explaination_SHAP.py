import shap
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests

all_songs = pd.read_csv("all_songs_encoded.csv")
genre_enc = joblib.load("genre_encoder.pkl")
language_enc = joblib.load("language_encoder.pkl")
type_enc = joblib.load("type_encoder.pkl")

feature_names = [
    "Genre",
    "Language",
    "Type",
]

def fetch_group_members(group_id):
    url = f"http://localhost:5000/api/export/group_member/{group_id}"
    res = requests.get(url)

    if res.status_code != 200:
        raise Exception("Failed to fetch group members")

    data = res.json()
    return data["members"]

def set_files(user_id):

    global model, explainer

    model_path = os.path.join("model", user_id, "song_recommendation_model.pkl")
    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)

def filter_songs_for_user():

    candidate_songs = all_songs.copy().reset_index(drop=True)

    inputs_df = candidate_songs[
        ["genre_encoded", "language_encoded", "type_encoded"]
    ]

    return candidate_songs, inputs_df

def score_songs(candidate_songs, inputs_df):
    probs_all = model.predict_proba(inputs_df)

    # Use confidence as score
    scores = probs_all.max(axis=1)

    candidate_songs["score"] = scores
    return candidate_songs

def decode_features(row):
    return {
        "Genre": genre_enc.inverse_transform([int(row[0])])[0],
        "Language": language_enc.inverse_transform([int(row[1])])[0],
        "Type": type_enc.inverse_transform([int(row[2])])[0],
    }

def get_user_song_scores(user_id):
    set_files(user_id)

    candidate_songs, inputs_df = filter_songs_for_user()
    scored = score_songs(candidate_songs, inputs_df)

    return scored[["song_id", "score"]]

def explain_song_for_user(user_id, song_row):
    set_files(user_id)

    X = pd.DataFrame([[
        song_row["genre_encoded"],
        song_row["language_encoded"],
        song_row["type_encoded"]
    ]], columns=["genre_encoded", "language_encoded", "type_encoded"])

    shap_values = explainer(X)

    # Use predicted class
    probs = model.predict_proba(X)[0]
    pred_class = np.argmax(probs)

    values = shap_values.values[0, :, pred_class]

    decoded = decode_features(X.iloc[0].values)

    explanation = {
        "helped": [],
        "hurt": []
    }

    for fname, val in zip(feature_names, values):
        label = decoded[fname]
        if val > 0:
            explanation["helped"].append(f"{fname}: {label} ({val:.3f})")
        elif val < 0:
            explanation["hurt"].append(f"{fname}: {label} ({val:.3f})")

    return explanation


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

def recommend_group_playlist(group_id, top_n=10, alpha=0.7):
    members = fetch_group_members(group_id)

    all_user_scores = []

    for uid in members:
        scores = get_user_song_scores(uid)
        if scores is not None:
            all_user_scores.append(scores)

    if len(all_user_scores) < 2:
        print("Not enough user data for group recommendation")
        return []

    group_scores = aggregate_group_scores(all_user_scores, alpha)

    return group_scores.head(top_n)

def explain_group_song(group_id, song_id, alpha=0.7):
    members = fetch_group_members(group_id)

    explanations = []

    song_row = all_songs[all_songs["song_id"] == song_id].iloc[0]

    for uid in members:
        try:
            exp = explain_song_for_user(uid, song_row)
            explanations.append(exp)
        except:
            continue

    helped = {}
    hurt = {}

    for exp in explanations:
        for h in exp["helped"]:
            helped[h] = helped.get(h, 0) + 1
        for h in exp["hurt"]:
            hurt[h] = hurt.get(h, 0) + 1

    return {
        "helped": sorted(helped, key=helped.get, reverse=True),
        "hurt": sorted(hurt, key=hurt.get, reverse=True)
    }


# Run case test for explaination
# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

# Group Id
# 6932fb410eace21333076758

# Weather Conditions
# Clouds
# Rain

# Time of Day
# day
# night

group_results = recommend_group_playlist(
    group_id="6932fb410eace21333076758",
    top_n=10,
    alpha=0.7
) 

for i, row in group_results.iterrows():
    print(f"Song ID: {row['song_id']} | Final Score: {row['final_score']:.4f}")

top_song_id = group_results.iloc[0]["song_id"]

group_explanation = explain_group_song(
    group_id="6932fb410eace21333076758",
    song_id=top_song_id
)

print("GROUP EXPLANATION")
print("Helped by:", group_explanation["helped"])
print("Hurt by:", group_explanation["hurt"])

##for r in results:
##    print(f"\nSong ID: {r['song_id']} | Score: {r['score']:.3f}")
##    print("Helped by:", r["explanation"]["helped"])
##    print("Hurt by:", r["explanation"]["hurt"])
