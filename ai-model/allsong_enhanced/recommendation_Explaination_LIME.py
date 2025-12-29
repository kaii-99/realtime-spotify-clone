from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import joblib

# Load your model and data
model = joblib.load("song_recommendation_model.pkl")
all_songs = pd.read_csv("all_songs_encoded.csv")
processed_data = pd.read_csv("processed_data.csv")
processed_data.columns = processed_data.columns.str.strip()
user_enc = joblib.load("user_encoder.pkl")
weather_enc = joblib.load("weather_encoder.pkl")
time_enc = joblib.load("timeofday_encoder.pkl")

# Features (drop user_id if you want cleaner explanations)
feature_names = ["user_id_encoded", "genre_encoded", "language_encoded", "type_encoded", "weather_encoded", "timeOfDay_encoded"]

def safe_transform(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0

def encode_user_inputs(user_id, weather, timeOfDay):
    return (
        safe_transform(user_enc, user_id),
        safe_transform(weather_enc, weather),
        safe_transform(time_enc, timeOfDay)
    )

def filter_songs_for_user(user_id, weather, timeOfDay):
    user_id_enc, weather_enc_val, time_enc_val = encode_user_inputs(user_id, weather, timeOfDay)

    # For simplicity, no top-genre filtering
    inputs = []
    candidate_songs = []

    for _, song in all_songs.iterrows():
        candidate_songs.append(song)
        inputs.append([
            user_id_enc,
            song["genre_encoded"],
            song["language_encoded"],
            song["type_encoded"],
            weather_enc_val,
            time_enc_val
        ])

    inputs_df = pd.DataFrame(inputs, columns=feature_names)
    return candidate_songs, inputs_df

# Wrapper function for LIME
def predict_proba_wrapper(X):
    # X is a numpy array
    X_df = pd.DataFrame(X, columns=feature_names)
    return model.predict_proba(X_df)  # returns [[prob_0, prob_1], ...]

def explain_with_lime(inputs_df, candidate_songs, top_n=5):
    categorical_features = [0, 1, 2, 3, 4, 5]  # indices of all encoded features
    categorical_names = {
        0: ["user1", "user2", "user3"],  # optional, can leave as str
        1: ["Indie", "Pop", "Rock", "Electronic", "Jazz", "Ballad", "Hip-Hop"],
        2: ["English", "Japanese", "Mandarin"],
        3: ["Normal", "Instrumental"],
        4: ["Clouds", "Rain"],
        5: ["day", "night"]
    }

    explainer = LimeTabularExplainer(
        training_data=inputs_df.values,
        feature_names=feature_names,
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        mode="classification"
    )

    # Score songs
    probs = model.predict_proba(inputs_df)[:, 1]
    top_indices = probs.argsort()[-top_n:][::-1]

    results = []

    for idx in top_indices:
        song = candidate_songs[idx]
        instance = inputs_df.iloc[idx].values

        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_proba_wrapper,
            num_features=len(feature_names)
        )

        # Convert explanation to dict
        helped = [f for f, v in exp.as_list() if v > 0]
        hurt = [f for f, v in exp.as_list() if v < 0]

        results.append({
            "song_id": song["song_id"],
            "score": float(inputs_df.iloc[idx]["score"]),
            "explanation": {
                "helped": helped,
                "hurt": hurt
            }
        })

        # Optional: plot bar
        exp.show_in_notebook(show_table=True)

    return results

# Usage
candidate_songs, inputs_df = filter_songs_for_user(
    user_id="690f6586822864ba799ef3a3",
    weather="Rain",
    timeOfDay="night"
)

results = explain_with_lime(inputs_df, candidate_songs, top_n=5)

for r in results:
    print(f"\nSong ID: {r['song_id']} | Score: {r['score']:.3f}")
    print("Helped by:", r["explanation"]["helped"])
    print("Hurt by:", r["explanation"]["hurt"])
