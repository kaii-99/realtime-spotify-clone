import pandas as pd

df = pd.read_csv("listening_data.csv")

# Drop unnecessary columns
df = df[["user_id", "song_id", "genre", "language", "type"]]

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

genre_enc = LabelEncoder()
language_enc = LabelEncoder()
type_enc = LabelEncoder()

df["genre_encoded"] = genre_enc.fit_transform(df["genre"])
df["language_encoded"] = language_enc.fit_transform(df["language"])
df["type_encoded"] = type_enc.fit_transform(df["type"])

df.to_csv("processed_data.csv", index=False)
print("✅ Processed data saved to processed_data.csv")
