import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
import os

# ==== Dataset Class ====
class MusicDataset(Dataset):
    def __init__(self, data_file, user_enc_file, weather_enc_file, time_enc_file, song_enc_file):
        self.df = pd.read_csv(data_file)
        self.user_enc = joblib.load(user_enc_file)
        self.weather_enc = joblib.load(weather_enc_file)
        self.time_enc = joblib.load(time_enc_file)
        self.song_enc = joblib.load(song_enc_file)
        
        # Encode inputs
        self.df['user_id'] = self.user_enc.transform(self.df['user_id'])
        self.df['song_id'] = self.song_enc.transform(self.df['song_id'])
        self.df['weather'] = self.weather_enc.transform(self.df['weather'])
        self.df['timeOfDay'] = self.time_enc.transform(self.df['timeOfDay'])
        
        # Label = 1 (listened) for supervised ranking
        self.df['label'] = 1

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user = torch.tensor(row['user_id'], dtype=torch.long)
        song = torch.tensor(row['song_id'], dtype=torch.long)
        weather = torch.tensor(row['weather'], dtype=torch.float)
        time = torch.tensor(row['timeOfDay'], dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.float)
        return user, song, weather, time, label

# ==== Model ====
class ContextAwareRecommender(nn.Module):
    def __init__(self, num_users, num_songs, embedding_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.song_emb = nn.Embedding(num_songs, embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim*2 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user, song, weather, time):
        u = self.user_emb(user)
        s = self.song_emb(song)
        x = torch.cat([u, s, weather.unsqueeze(1), time.unsqueeze(1)], dim=1)
        out = self.fc(x)
        return out.squeeze()

# ==== Load data ====
USER_ID = "690f6586822864ba799ef3a3"
processed_folder = os.path.join("processed_data", USER_ID)
data_file = os.path.join(processed_folder, f"{USER_ID}_processed_data.csv")

user_enc_file = os.path.join(processed_folder, "user_encoder.pkl")
weather_enc_file = os.path.join(processed_folder, "weather_encoder.pkl")
time_enc_file = os.path.join(processed_folder, "timeofday_encoder.pkl")
song_enc_file = "song_encoder.pkl"

dataset = MusicDataset(data_file, user_enc_file, weather_enc_file, time_enc_file, song_enc_file)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ==== Instantiate model ====
num_users = len(joblib.load(user_enc_file).classes_)
num_songs = len(joblib.load(song_enc_file).classes_)

model = ContextAwareRecommender(num_users=num_users, num_songs=num_songs)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== Training loop (example) ====
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for user, song, weather, time, label in dataloader:
        optimizer.zero_grad()
        pred = model(user, song, weather, time)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

# ==== Save model ====
model_folder = os.path.join("model", USER_ID)
os.makedirs(model_folder, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_folder, "context_model.pth"))
print("✅ Context-aware model saved!")
