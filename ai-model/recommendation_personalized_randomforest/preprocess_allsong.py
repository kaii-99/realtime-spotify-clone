import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load raw data
df = pd.read_csv("allsong_data.csv")

# Drop unnecessary columns and rename _id -> song_id
df = df[["_id", "genre", "language", "type"]].rename(columns={"_id": "song_id"})

# Create encoders
song_enc = LabelEncoder()
genre_enc = LabelEncoder()
language_enc = LabelEncoder()
type_enc = LabelEncoder()

# Fit encoders
df["song_id_encoded"] = song_enc.fit_transform(df["song_id"])
df["genre_encoded"] = genre_enc.fit_transform(df["genre"])
df["language_encoded"] = language_enc.fit_transform(df["language"])
df["type_encoded"] = type_enc.fit_transform(df["type"])

# SAVE ENCODERS (CRITICAL STEP)
joblib.dump(song_enc, "song_encoder.pkl")
joblib.dump(genre_enc, "genre_encoder.pkl")
joblib.dump(language_enc, "language_encoder.pkl")
joblib.dump(type_enc, "type_encoder.pkl")

# Save processed dataset
df.to_csv("all_songs_encoded.csv", index=False)
print("Processed data saved to all_songs_encoded.csv")
