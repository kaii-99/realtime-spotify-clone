import os
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import shap
import json
import requests

BASE_API_URL = os.environ.get("BACKEND_URL")

# Model
class RecommenderNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def fetch_group_members(group_id):
    url = f"{BASE_API_URL}/api/export/group_member/{group_id}"
    res = requests.get(url)

    if res.status_code != 200:
        raise Exception("Failed to fetch group members")

    data = res.json()
    return data["members"]

def load_user_model(user_id):
    model_path = f"group_playlist_randomforest/model/{user_id}/{user_id}_personalized_model_pytorch.pth"
    song_map_path = f"group_playlist_randomforest/model/{user_id}/{user_id}_song_mapping.json"
    with open(song_map_path, "r") as f:
        idx_to_song = {int(k): v for k, v in json.load(f).items()}

    num_songs = len(idx_to_song)

    model = RecommenderNN(
        input_dim=3,
        hidden_dim=64,
        output_dim=num_songs
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model, idx_to_song

#def user_scores_and_shap(model, all_songs_df):
#    X = torch.tensor(
#        all_songs_df[["genre_encoded", "language_encoded", "type_encoded"]].values,
#        dtype=torch.float32
#    )
#
#    with torch.no_grad():
#        logits = model(X)
#        probs = torch.softmax(logits, dim=1)
#        scores, pred_class = probs.max(dim=1) 
#
#    scores = scores.numpy()
#    pred_class = pred_class.numpy()
#
#    # SHAP 
#    background = X[:min(50, len(X))]
#    explainer = shap.DeepExplainer(model, background)
#
#    shap_vals = explainer.shap_values(X)
#
#    per_sample_shap = np.array([
#        shap_vals[c][i] for i, c in enumerate(pred_class)
#    ])
#
#    shap_feature_mean = np.mean(np.abs(per_sample_shap), axis=0)
#
#    return scores, shap_feature_mean

def group_recommendation(user_ids, all_songs_df, alpha=0.7, top_n=10):
    score_list = []

    for uid in user_ids:
        model, song_map = load_user_model(uid)
        X = torch.tensor(all_songs_df[["genre_encoded", "language_encoded", "type_encoded"]].values, dtype=torch.float32)
        with torch.no_grad():
            scores = torch.softmax(model(X), dim=1).max(dim=1).values.numpy()
        score_list.append(scores)

    scores_np = np.vstack(score_list)
    avg_score = scores_np.mean(axis=0)
    min_score = scores_np.min(axis=0)
    final_score = alpha * avg_score + (1 - alpha) * min_score

    result = all_songs_df.copy()
    result["final_score"] = final_score

    return result.sort_values("final_score", ascending=False).head(top_n)

def group_recommendation_DL(group_id):
    all_songs = pd.read_csv("group_playlist_randomforest/all_songs_encoded.csv")
    all_songs.columns = all_songs.columns.str.strip()

    members = fetch_group_members(group_id)

    top_songs = group_recommendation(
        user_ids=members,
        all_songs_df=all_songs,
        alpha=0.7,
        top_n=10
    )

    for _, row in top_songs.iterrows():
        print(f"Song: {row['song_id']}, Score: {row['final_score']:.4f}")

    return top_songs