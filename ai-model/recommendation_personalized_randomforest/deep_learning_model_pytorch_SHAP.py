import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import shap

def load_user_data(user_id):
    """Load processed user data and encoders from user folder"""
    USER_FOLDER = os.path.join("processed_data", user_id)
    
    # Processed data
    data_file = os.path.join(USER_FOLDER, f"{user_id}_processed_data.csv")
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()

    # Encoders
    weather_enc = joblib.load(os.path.join(USER_FOLDER, "weather_encoder.pkl"))
    time_enc = joblib.load(os.path.join(USER_FOLDER, "timeofday_encoder.pkl"))
    genre_enc = joblib.load("genre_encoder.pkl")
    language_enc = joblib.load("language_encoder.pkl")
    type_enc = joblib.load("type_encoder.pkl")
    song_enc = joblib.load("song_encoder.pkl")

    song_id_to_idx = {song: idx for idx, song in enumerate(sorted(df["song_id_encoded"].unique()))}
    df["song_idx"] = df["song_id_encoded"].map(song_id_to_idx)
    
    return df, weather_enc, time_enc, genre_enc, language_enc, type_enc, song_enc

class UserDataset(Dataset):
    def __init__(self, df):
        self.X = df[["genre_encoded", "language_encoded", "type_encoded", "weather_encoded", "timeOfDay_encoded"]].values
        self.y = df["song_idx"].values

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)  # For classification/ranking
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RecommenderNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, output_dim)  # Output: # of songs
    
    def forward(self, x):
        return self.out(self.fc(x))

def train_model(df, num_songs, epochs=30, batch_size=8, lr=0.001):
    dataset = UserDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RecommenderNN(input_dim=5, hidden_dim=64, output_dim=num_songs)
    criterion = nn.CrossEntropyLoss()  # Treat as classification among songs
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    return model

def _extract_deep_shap_for_preds(shap_values, pred_classes):
    # SHAP output shape varies by version/model; normalize to per-sample, per-feature values.
    if isinstance(shap_values, list):
        # list[class_id] -> (n_samples, n_features)
        return np.array([shap_values[c][i] for i, c in enumerate(pred_classes)])

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return np.array([arr[i, :, c] for i, c in enumerate(pred_classes)])

    if arr.ndim == 2:
        # Already (n_samples, n_features)
        return arr

    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

def _apply_context_rerank(scores_df, processed_df, weather_val, time_val):
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

def recommend_songs(
    model,
    all_songs_df,
    weather_enc_val,
    time_enc_val,
    song_encoded_to_idx,
    user_id=None,
    exclude_listened=False,
    context_rerank=False,
    top_n=5,
    explain=False
):
    """
    all_songs_df: song features with genre_encoded, language_encoded, type_encoded
    """
    inputs = []
    for _, song in all_songs_df.iterrows():
        inputs.append([
            song["genre_encoded"],
            song["language_encoded"],
            song["type_encoded"],
            weather_enc_val,
            time_enc_val
        ])
    
    X = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X)
        probs = torch.softmax(scores, dim=1).numpy()
    
    all_songs_df = all_songs_df.copy()
    class_indices = all_songs_df["song_id_encoded"].map(song_encoded_to_idx).fillna(-1).astype(int).to_numpy()
    row_indices = np.arange(len(all_songs_df))
    valid_mask = class_indices >= 0
    # Fallback for songs outside user-specific class mapping.
    song_scores = probs.max(axis=1)
    song_scores[valid_mask] = probs[row_indices[valid_mask], class_indices[valid_mask]]
    all_songs_df["score"] = song_scores

    if context_rerank and user_id is not None:
        processed_path = os.path.join("processed_data", user_id, f"{user_id}_processed_data.csv")
        if os.path.exists(processed_path):
            processed_df = pd.read_csv(processed_path)
            processed_df.columns = processed_df.columns.str.strip()
            all_songs_df = _apply_context_rerank(all_songs_df, processed_df, weather_enc_val, time_enc_val)

    if exclude_listened and user_id is not None:
        history_path = os.path.join("listening_history_data", f"{user_id}_listening_data.csv")
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            if "song_id" in history_df.columns:
                listened_song_ids = set(history_df["song_id"].astype(str))
                all_songs_df = all_songs_df[~all_songs_df["song_id"].astype(str).isin(listened_song_ids)]

    top_songs = (
        all_songs_df
        .drop_duplicates(subset=["song_id"])
        .sort_values("score", ascending=False)
        .head(top_n)
        .copy()
    )
    
    if explain:
        # Use a small subset as background
        background = X[:min(50, len(X))]
        explainer = shap.DeepExplainer(model, background)
        top_indices = top_songs.index.to_list()
        X_top = X[top_indices]
        shap_values = explainer.shap_values(X_top)

        with torch.no_grad():
            pred_classes = torch.softmax(model(X_top), dim=1).argmax(dim=1).numpy()

        per_sample_shap = _extract_deep_shap_for_preds(shap_values, pred_classes)
        top_songs["shap_values"] = [sv.tolist() for sv in per_sample_shap]
    
    return top_songs

if __name__ == "__main__":
    # User-specific
    # Run case test for explaination
    # testuser1 -> 690f61b5822864ba799ef31d
    # testuser2 -> 690f6586822864ba799ef3a3
    # testuser3 -> 690f66bf822864ba799ef3fc
    USER_ID = "690f66bf822864ba799ef3fc"

    # Load data/encoders
    df, weather_enc, time_enc, genre_enc, language_enc, type_enc, song_enc = load_user_data(USER_ID)
    all_songs = pd.read_csv("all_songs_encoded.csv")
    all_songs.columns = all_songs.columns.str.strip()

    # Use saved production model + mapping so SHAP run matches server behavior.
    model_path = os.path.join("model", USER_ID, f"{USER_ID}_personalized_model_pytorch.pth")
    map_path = os.path.join("model", USER_ID, f"{USER_ID}_song_mapping.json")
    if os.path.exists(model_path) and os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            idx_to_song_encoded = json.load(f)
        song_encoded_to_idx = {int(v): int(k) for k, v in idx_to_song_encoded.items()}
        num_songs = len(idx_to_song_encoded)
        model = RecommenderNN(input_dim=5, hidden_dim=64, output_dim=num_songs)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        num_songs = len(df["song_idx"].unique())
        model = train_model(df, num_songs=num_songs, epochs=20)
        song_encoded_to_idx = {song: idx for idx, song in enumerate(sorted(df["song_id_encoded"].unique()))}

    # Get all unique weather/time combinations from user's history
    unique_conditions = df[["weather_encoded", "timeOfDay_encoded"]].drop_duplicates()

    for _, row in unique_conditions.iterrows():
        weather_val = row["weather_encoded"]
        time_val = row["timeOfDay_encoded"]

        # Decode for display (optional)
        weather_label = weather_enc.inverse_transform([weather_val])[0]
        time_label = time_enc.inverse_transform([time_val])[0]

        # Get top songs with SHAP explanations
        top_songs = recommend_songs(
            model,
            all_songs,
            weather_val,
            time_val,
            song_encoded_to_idx=song_encoded_to_idx,
            user_id=USER_ID,
            exclude_listened=True,
            context_rerank=True,
            top_n=5,
            explain=True
        )

        print(f"\nTop songs for Weather: {weather_label}, Time: {time_label}")
        print(top_songs[["song_id", "score"]])

        # Print SHAP feature contributions
        feature_names = ["genre_encoded","language_encoded","type_encoded","weather_encoded","timeOfDay_encoded"]
        for _, song_row in top_songs.iterrows():
            print(f"\nSong {song_row['song_id']}:")
            for name, val in zip(feature_names, song_row["shap_values"]):
                print(f"  {name}: {val:.3f}")
