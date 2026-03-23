import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

INPUT_FOLDER = "listening_history_data"
OUTPUT_ROOT = "processed_data"

# Load global encoders once
song_enc = joblib.load("song_encoder.pkl")
genre_enc = joblib.load("genre_encoder.pkl")
language_enc = joblib.load("language_encoder.pkl")
type_enc = joblib.load("type_encoder.pkl")

# Loop through all user files
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith("_listening_data.csv"):
        continue

    USER_ID = filename.replace("_listening_data.csv", "")
    INPUT_FILE = os.path.join(INPUT_FOLDER, filename)
    OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, USER_ID)
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, f"{USER_ID}_processed_data.csv")

    print(f"\n▶ Processing user: {USER_ID}")

    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load user data
    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("No data, skipped")
        continue

    # Keep only non-mood columns
    df = df[[
        "song_id",
        "genre",
        "language",
        "type"
    ]]

    # Encode
    df["song_id_encoded"] = song_enc.transform(df["song_id"])
    df["genre_encoded"] = genre_enc.transform(df["genre"])
    df["language_encoded"] = language_enc.transform(df["language"])
    df["type_encoded"] = type_enc.transform(df["type"])

    # Drop text columns
    df = df.drop(columns=[
        "song_id",
        "genre",
        "language",
        "type"
    ])

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved → {OUTPUT_FILE}")
    print(f"   Rows: {len(df)}")

print("\n🎉 All users processed successfully")