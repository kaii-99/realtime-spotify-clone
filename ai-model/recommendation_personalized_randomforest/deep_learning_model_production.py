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
    USER_FOLDER = os.path.join("recommendation_personalized_randomforest/processed_data", user_id)
    
    # Processed data
    data_file = os.path.join(USER_FOLDER, f"{user_id}_processed_data.csv")
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()

    # Encoders
    weather_enc = joblib.load(os.path.join(USER_FOLDER, "weather_encoder.pkl"))
    time_enc = joblib.load(os.path.join(USER_FOLDER, "timeofday_encoder.pkl"))
    genre_enc = joblib.load("recommendation_personalized_randomforest/genre_encoder.pkl")
    language_enc = joblib.load("recommendation_personalized_randomforest/language_encoder.pkl")
    type_enc = joblib.load("recommendation_personalized_randomforest/type_encoder.pkl")
    song_enc = joblib.load("recommendation_personalized_randomforest/song_encoder.pkl")

    song_id_to_idx = {song: idx for idx, song in enumerate(sorted(df["song_id_encoded"].unique()))}
    df["song_idx"] = df["song_id_encoded"].map(song_id_to_idx)
    
    return df, weather_enc, time_enc, genre_enc, language_enc, type_enc, song_enc, song_id_to_idx

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

def recommend_songs(model, all_songs_df, weather_enc_val, time_enc_val, top_n=5):
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
    
    # Add scores to the DataFrame
    all_songs_df = all_songs_df.copy()  # avoid SettingWithCopyWarning
    all_songs_df["score"] = probs.max(axis=1)
    
    # Get top songs
    top_songs = all_songs_df.sort_values("score", ascending=False).head(top_n)
    
    return top_songs

# User-specific
USER_ID = "690f6586822864ba799ef3a3"

# Load data
df, weather_enc, time_enc, genre_enc, language_enc, type_enc, song_enc, song_id_to_idx = load_user_data(USER_ID)
all_songs = pd.read_csv("recommendation_personalized_randomforest/all_songs_encoded.csv")
all_songs.columns = all_songs.columns.str.strip()

# Train personalized model first
num_songs = len(df["song_idx"].unique())
model = train_model(df, num_songs=num_songs, epochs=20)

# Get all unique weather/time combinations from user's history
unique_conditions = df[["weather_encoded", "timeOfDay_encoded"]].drop_duplicates()

for _, row in unique_conditions.iterrows():
    weather_val = row["weather_encoded"]
    time_val = row["timeOfDay_encoded"]

    # Decode for display (optional)
    weather_label = weather_enc.inverse_transform([weather_val])[0]
    time_label = time_enc.inverse_transform([time_val])[0]

    top_songs = recommend_songs(model, all_songs, weather_val, time_val, top_n=5)
    print(f"\nTop songs for Weather: {weather_label}, Time: {time_label}")
    print(top_songs[["song_id", "score"]])

## Save the model
#torch.save(model.state_dict(), f"model/{USER_ID}/{USER_ID}_personalized_model.pth")
#
## Save song-id mapping 
## Convert numpy.int64 -> normal int (or str)
#clean_mapping = {str(k): int(v) for k, v in song_id_to_idx.items()}
#with open(f"model/{USER_ID}/{USER_ID}_song_mapping.json", "w") as f:
#    json.dump(clean_mapping, f)