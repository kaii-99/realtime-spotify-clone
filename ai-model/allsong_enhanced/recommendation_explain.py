import shap
import numpy as np
import joblib
import pandas as pd

model = joblib.load("song_recommendation_model.pkl")
all_songs = pd.read_csv("all_songs_encoded.csv")
user_enc = joblib.load("user_encoder.pkl")
weather_enc = joblib.load("weather_encoder.pkl")
time_enc = joblib.load("timeofday_encoder.pkl")

explainer = shap.TreeExplainer(model)

feature_names = [
    "User ID",
    "Genre",
    "Language",
    "Type",
    "Weather",
    "Time of Day"
]

def safe_transform(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0


def encode_user_inputs(user_id, weather, timeOfDay):
    return (
        safe_transform(user_enc, user_id),
        safe_transform(weather_enc, weather),
        safe_transform(time_enc, timeOfDay)
    )


def recommend_and_explain(user_id, weather, timeOfDay, top_n=5):

    user_id_enc, weather_enc_val, time_enc_val = encode_user_inputs(
        user_id, weather, timeOfDay
    )

    inputs = []

    for _, song in all_songs.iterrows():
        inputs.append([
            user_id_enc,
            song["genre_encoded"],
            song["language_encoded"],
            song["type_encoded"],
            weather_enc_val,
            time_enc_val
        ])

    

    # 1️⃣ Get prediction probabilities
    # Don't use -> This based on overall (Great for general use not for personal recommendation)
    # inputs = np.array(inputs)
    # probs = model.predict_proba(inputs)

    inputs_df = pd.DataFrame(inputs, columns=[
        "user_id_encoded",
        "genre_encoded",
        "language_encoded",
        "type_encoded",
        "weather_encoded",
        "timeOfDay_encoded"
    ])
    probs = model.predict_proba(inputs_df)
    song_scores = probs[:, 1]  # probability of LIKE

    all_songs["score"] = song_scores

    # 2️⃣ Get top N songs
    top_songs = all_songs.sort_values("score", ascending=False).head(top_n)

    print("\nTOP RECOMMENDATIONS WITH EXPLANATIONS:\n")

    # 3️⃣ Explain each recommended song
    for _, song in top_songs.iterrows():

        index = song.name  # row index in dataframe
        shap_values = explainer(inputs_df.iloc[index:index+1])
        predicted_class = np.argmax(probs[index])

        explanation = shap.Explanation(
            values=shap_values.values[0, :, predicted_class],
            base_values=shap_values.base_values[0, predicted_class],
            data=inputs[index],
            feature_names=[
                "User ID",
                "Genre",
                "Language",
                "Type",
                "Weather",
                "Time of Day"
            ]
        )

        print(f"Song ID: {song['song_id']} | Score: {song['score']:.3f}")
        shap.plots.bar(explanation)

# Run case test for explaination
# testuser1 -> 690f61b5822864ba799ef31d
# testuser2 -> 690f6586822864ba799ef3a3
# testuser3 -> 690f66bf822864ba799ef3fc

# Weather Conditions
# Clouds
# Rain

# Time of Day
# day
# night

recommend_and_explain(
    user_id="690f6586822864ba799ef3a3",
    weather="Rain",
    timeOfDay="night"
)