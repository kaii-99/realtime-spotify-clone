import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your listening data
df = pd.read_csv("listening_data.csv")

# 1. Keep only relevant columns
df = df[[
    "user_id",
    "song_id",
    "genre",
    "language",
    "type",
    "weather",
    "timeOfDay"
]]

song_enc = joblib.load("song_encoder.pkl")
genre_enc = joblib.load("genre_encoder.pkl")
language_enc = joblib.load("language_encoder.pkl")
type_enc = joblib.load("type_encoder.pkl")

df["song_id_encoded"] = song_enc.transform(df["song_id"])
df["genre_encoded"] = genre_enc.transform(df["genre"])
df["language_encoded"] = language_enc.transform(df["language"])
df["type_encoded"] = type_enc.transform(df["type"])

# 2. Remove rows missing weather or timeOfDay
df = df.dropna(subset=["weather", "timeOfDay"])

# Optional: Remove empty strings too
df = df[(df["weather"] != "") & (df["timeOfDay"] != "")]

# 3. Encode categorical columns
user_enc = LabelEncoder()
weather_enc = LabelEncoder()
time_enc = LabelEncoder()

df["user_id_encoded"] = user_enc.fit_transform(df["user_id"])
df["weather_encoded"] = weather_enc.fit_transform(df["weather"])
df["timeOfDay_encoded"] = time_enc.fit_transform(df["timeOfDay"])

# 4. Drop original text columns (optional but recommended for ML)
df = df.drop(columns=[
    "user_id",
    "song_id",
    "genre",
    "language",
    "type",
    "weather",
    "timeOfDay"
])

# ✅ 5. Save cleaned dataset
df.to_csv("processed_data.csv", index=False)

joblib.dump(user_enc, "user_encoder.pkl")
joblib.dump(weather_enc, "weather_encoder.pkl")
joblib.dump(time_enc, "timeofday_encoder.pkl")


print("✅ Clean ML dataset created: processed_data.csv")
print("Rows:", len(df))
