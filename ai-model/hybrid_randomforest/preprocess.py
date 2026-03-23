import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

#USER_ID = "690f6586822864ba799ef3a3"

#INPUT_FOLDER = "listening_history_data"
#OUTPUT_FOLDER = os.path.join("processed_data", USER_ID)

#INPUT_FILE = os.path.join(INPUT_FOLDER, f"{USER_ID}_listening_data.csv")
#OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, f"{USER_ID}_processed_data.csv")

#all users
USER_IDS = {
    "testuser1": "690f61b5822864ba799ef31d",
    "testuser2": "690f6586822864ba799ef3a3",
    "testuser3": "690f66bf822864ba799ef3fc"
}

INPUT_FOLDER = "listening_history_data"
BASE_OUTPUT_FOLDER = "processed_data"

# Load shared encoders
song_enc = joblib.load("song_encoder.pkl")
genre_enc = joblib.load("genre_encoder.pkl")
language_enc = joblib.load("language_encoder.pkl")
type_enc = joblib.load("type_encoder.pkl")

for username, user_id in USER_IDS.items():
    INPUT_FILE = os.path.join(INPUT_FOLDER, f"{user_id}_listening_data.csv")
    OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, user_id)
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, f"{user_id}_processed_data.csv")

    # Create folder if not exists
    if not os.path.exists(INPUT_FILE):
        print(f"File not found for {username}: {INPUT_FILE}")
        continue

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # 1. Keep only relevant columns
    df = df[[
        "song_id",
        "genre",
        "language",
        "type",
        "weather",
        "timeOfDay"
    ]]

    # 2. Encode fixed categorical columns
    df["song_id_encoded"] = song_enc.transform(df["song_id"])
    df["genre_encoded"] = genre_enc.transform(df["genre"])
    df["language_encoded"] = language_enc.transform(df["language"])
    df["type_encoded"] = type_enc.transform(df["type"])

    # 3. Remove rows missing weather or timeOfDay
    df = df.dropna(subset=["weather", "timeOfDay"])
    df = df[(df["weather"] != "") & (df["timeOfDay"] != "")]

    # 4. Encode contextual features (per user)
    weather_enc = LabelEncoder()
    time_enc = LabelEncoder()

    df["weather_encoded"] = weather_enc.fit_transform(df["weather"])
    df["timeOfDay_encoded"] = time_enc.fit_transform(df["timeOfDay"])

    # 5. Drop original text columns
    df = df.drop(columns=[
        "song_id",
        "genre",
        "language",
        "type",
        "weather",
        "timeOfDay"
    ])

    # 6. Save processed dataset + encoders
    df.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(weather_enc, os.path.join(OUTPUT_FOLDER, "weather_encoder.pkl"))
    joblib.dump(time_enc, os.path.join(OUTPUT_FOLDER, "timeofday_encoder.pkl"))

    print(f"Processed data saved for {username}")
    print(f"Rows: {len(df)}\n")