import shap
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

all_songs = pd.read_csv("all_songs_encoded.csv")
genre_enc = joblib.load("genre_encoder.pkl")
language_enc = joblib.load("language_encoder.pkl")
type_enc = joblib.load("type_encoder.pkl")

feature_names = [
    "Genre",
    "Language",
    "Type",
    "Weather",
    "Time of Day"
]

def set_files(user_id):

    global model, processed_data, weather_enc, time_enc, explainer

    USER_ID = user_id

    MODEL_FOLDER = os.path.join("model", USER_ID)
    PROCESSED_FOLDER = os.path.join("processed_data", USER_ID)

    model_file = os.path.join(MODEL_FOLDER, "song_recommendation_model.pkl")
    model = joblib.load(model_file)

    processed_data = pd.read_csv(os.path.join(PROCESSED_FOLDER, f"{USER_ID}_processed_data.csv"))
    processed_data.columns = processed_data.columns.str.strip()

    weather_enc = joblib.load(os.path.join(PROCESSED_FOLDER, "weather_encoder.pkl"))
    time_enc = joblib.load(os.path.join(PROCESSED_FOLDER, "timeofday_encoder.pkl"))

    # Try probability explanations first; fallback to raw for SHAP setups that don't support it.
    feature_cols = [
        "genre_encoded",
        "language_encoded",
        "type_encoded",
        "weather_encoded",
        "timeOfDay_encoded"
    ]
    background = processed_data[feature_cols].dropna().head(200)

    try:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
            model_output="probability"
        )
    except Exception:
        explainer = shap.TreeExplainer(model)

def _extract_positive_class_shap(shap_values):
    values = shap_values.values
    base_values = shap_values.base_values

    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    if isinstance(base_values, np.ndarray):
        if base_values.ndim == 2:
            base_value = base_values[0, 1]
        else:
            base_value = base_values[0]
    else:
        base_value = base_values

    return values, base_value

def safe_transform(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else 0

def encode_user_inputs(weather, timeOfDay):
    return (
        safe_transform(weather_enc, weather),
        safe_transform(time_enc, timeOfDay)
    )

def get_user_preferences():
    # Top 3 genres in user's history
    top_genres = processed_data["genre_encoded"].value_counts().head(3).index.tolist()
    preferred_langs = processed_data["language_encoded"].unique()
    return preferred_langs, top_genres

def filter_songs_for_user(weather, timeOfDay):
    weather_enc_val, time_enc_val = encode_user_inputs(weather, timeOfDay)
    preferred_langs, top_genres = get_user_preferences()

    # Filter songs
    candidate_songs = all_songs[
        all_songs["language_encoded"].isin(preferred_langs) &
        all_songs["genre_encoded"].isin(top_genres)
    ].copy()

    # Reset indices
    candidate_songs = candidate_songs.reset_index(drop=True)

    # Build input DataFrame
    inputs = []
    for _, song in candidate_songs.iterrows():
        inputs.append([
            song["genre_encoded"],
            song["language_encoded"],
            song["type_encoded"],
            weather_enc_val,
            time_enc_val
        ])
    inputs_df = pd.DataFrame(inputs, columns=[
        "genre_encoded",
        "language_encoded",
        "type_encoded",
        "weather_encoded",
        "timeOfDay_encoded"
    ])
    
    # Reset index for inputs_df too
    inputs_df = inputs_df.reset_index(drop=True)

    return candidate_songs, inputs_df

def score_songs(candidate_songs, inputs_df):
    # probability of like
    probs = model.predict_proba(inputs_df)[:, 1]
    candidate_songs["score"] = probs
    return candidate_songs

def decode_features(row):
    return {
        "Genre": genre_enc.inverse_transform([int(row[0])])[0],
        "Language": language_enc.inverse_transform([int(row[1])])[0],
        "Type": type_enc.inverse_transform([int(row[2])])[0],
        "Weather": weather_enc.inverse_transform([int(row[3])])[0],
        "Time of Day": time_enc.inverse_transform([int(row[4])])[0],
    }


def explain_top_songs(candidate_songs, inputs_df, top_n=5, show_plot=False):
    # Keep original indices so each explained row matches the correct model input row.
    top_songs = (
        candidate_songs
        .drop_duplicates(subset=["song_id"])
        .sort_values("score", ascending=False)
        .head(top_n)
    )

    results = []

    for row_idx, song in top_songs.iterrows():
        shap_values = explainer(inputs_df.iloc[row_idx:row_idx+1])
        values, base_value = _extract_positive_class_shap(shap_values)

        decoded_row = decode_features(inputs_df.iloc[row_idx].values)

        helped = []
        hurt = []

        for fname, val in zip(feature_names, values):
            label = decoded_row[fname]
            if val > 0:
                helped.append(f"{fname}: {label} ({val:.3f})")
            elif val < 0:
                hurt.append(f"{fname}: {label} ({val:.3f})")

        helped = sorted(
            helped,
            key=lambda x: abs(float(x.rsplit("(", 1)[1].rstrip(")"))),
            reverse=True
        )
        hurt = sorted(
            hurt,
            key=lambda x: abs(float(x.rsplit("(", 1)[1].rstrip(")"))),
            reverse=True
        )

        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=inputs_df.iloc[row_idx].values,
            feature_names=feature_names
        )

        # SHAP bar plot (optional for speed)
        if show_plot:
            shap.plots.bar(explanation, max_display=6, show=True)

        results.append({
            "song_id": song["song_id"],
            "score": float(song["score"]),
            "explanation": {
                "helped": helped,
                "hurt": hurt
            }
        })

    return results

def recommend_and_explain(user_id, weather, timeOfDay, top_n=5, show_plot=False):
    set_files(user_id)

    candidate_songs, inputs_df = filter_songs_for_user(weather, timeOfDay)
    
    if candidate_songs.empty:
        print("No candidate songs found for this user/preferences.")
        return []

    candidate_songs = score_songs(candidate_songs, inputs_df)
    results = explain_top_songs(candidate_songs, inputs_df, top_n, show_plot=show_plot)
    return results

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

if __name__ == "__main__":
    results = recommend_and_explain(
        user_id="690f61b5822864ba799ef31d",
        weather="Clouds",
        timeOfDay="night",
        show_plot=False
    )

    for r in results:
        print(f"\nSong ID: {r['song_id']} | Score: {r['score']:.3f}")
        print("Helped by:", r["explanation"]["helped"])
        print("Hurt by:", r["explanation"]["hurt"])
