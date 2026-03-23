import os
import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import json

def load_user_data(user_id):
    """Load processed user data and encoders from user folder"""
    USER_FOLDER = os.path.join("processed_data", user_id)
    
    # Processed data
    data_file = os.path.join(USER_FOLDER, f"{user_id}_processed_data.csv")
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()

    # Encoders
    genre_enc = joblib.load("genre_encoder.pkl")
    language_enc = joblib.load("language_encoder.pkl")
    type_enc = joblib.load("type_encoder.pkl")
    song_enc = joblib.load("song_encoder.pkl")

    song_id_to_idx = {song: idx for idx, song in enumerate(sorted(df["song_id_encoded"].unique()))}
    df["song_idx"] = df["song_id_encoded"].map(song_id_to_idx)
    
    return df, genre_enc, language_enc, type_enc, song_enc, song_id_to_idx

# Dataset
class UserDataset(Dataset):
    def __init__(self, df):
        self.X = df[[
            "genre_encoded",
            "language_encoded",
            "type_encoded",
            "weather_encoded",
            "timeOfDay_encoded"
        ]].values

        self.y = df["song_idx"].values

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    
 # Train Model
def train_user_model(df, num_songs, epochs=25, batch_size=16, lr=0.001):
    dataset = UserDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RecommenderNN(
        input_dim=5,
        hidden_dim=64,
        output_dim=num_songs
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    return model

def score_all_songs(model, all_songs_df):
    X = torch.tensor(
        all_songs_df[["genre_encoded", "language_encoded", "type_encoded","weather_encoded","timeOfDay_encoded"]].values,
        dtype=torch.float32
    )

    with torch.no_grad():
        scores = model(X).numpy()

    results = all_songs_df.copy()
    results["score"] = scores
    return results.sort_values("score", ascending=False)

# Loop All Users
PROCESSED_ROOT = "processed_data"
MODEL_ROOT = "model"

os.makedirs(MODEL_ROOT, exist_ok=True)

for USER_ID in os.listdir(PROCESSED_ROOT):
    USER_FOLDER = os.path.join(PROCESSED_ROOT, USER_ID)

    if not os.path.isdir(USER_FOLDER):
        continue

    INPUT_FILE = os.path.join(USER_FOLDER, f"{USER_ID}_processed_data.csv")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Missing data for {USER_ID}")
        continue

    print(f"\nTraining model for user: {USER_ID}")

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    # Build song index mapping
    song_ids = sorted(df["song_id_encoded"].unique())
    song_id_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    idx_to_song_id = {i: sid for sid, i in song_id_to_idx.items()}

    df["song_idx"] = df["song_id_encoded"].map(song_id_to_idx)

    num_songs = len(song_id_to_idx)

    model = train_user_model(df, num_songs)

    # Save
    USER_MODEL_FOLDER = os.path.join(MODEL_ROOT, USER_ID)
    os.makedirs(USER_MODEL_FOLDER, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), f"model/{USER_ID}/{USER_ID}_personalized_model_pytorch.pth")

    json_safe_mapping = {
        str(k): int(v) for k, v in idx_to_song_id.items()
    }
    
    mapping_path = os.path.join(
        MODEL_ROOT,
        USER_ID,
        f"{USER_ID}_song_mapping.json"
    )
    
    with open(mapping_path, "w") as f:
        json.dump(json_safe_mapping, f, indent=2)

    print(f"Model saved for {USER_ID}")