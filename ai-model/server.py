from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
from weather_detection.weather_mood import detect_mood_from_weather
from allsong_enhanced.recommend_mood_model import recommend_songs_for_user
from recommendation_personalized_randomforest.recommend_songs_for_user_deeplearning import recommend_songs_for_user_enhanced
from recommendation_personalized_randomforest.recommend_songs_for_user_machinelearning import recommend_songs_for_user_ml
from group_playlist_randomforest.recommend_songs_for_user_machinelearning import recommend_songs_for_group_ml
from group_playlist_randomforest.deep_learning_model_pytorch_production import group_recommendation_DL

app = Flask(__name__)

# Load the trained model data
model_data = joblib.load("allsong/trained_model.pkl")
similarity_matrix = model_data["similarity_matrix"]
song_id_to_index = model_data["song_id_to_index"]
index_to_song_id = model_data["index_to_song_id"]
song_features = model_data["song_features"]
encoder = model_data["encoder"]

@app.route("/weather-mood", methods=["GET"])
def weather_mood():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "city query parameter is required"}), 400

    data = detect_mood_from_weather(city)
    return jsonify(data)

@app.route("/recommendations", methods=["GET"])
def recommend():
    song_ids_param = request.args.get("songIds")
    city = request.args.get("city")

    if not song_ids_param:
        return jsonify({"error": "songIds query parameter is required"}), 400

    user_history = song_ids_param.split(",")

    # Detect Mood
    weather_data = detect_mood_from_weather(city) if city else None
    weather_genres = weather_data["recommended_genres"] if weather_data else []

    # Debug print
    print("🎯 Debug Weather Info:")
    print("City:", city)
    print("Detected Mood:", weather_data["mood"] if weather_data else None)
    print("Weather Condition:", weather_data["weather"] if weather_data else None)
    print("Recommended Genres:", weather_genres)

    # Initialize similarity scores
    sim_scores = np.zeros(len(song_id_to_index))

    for song_id in user_history:
        if song_id in song_id_to_index:
            idx = song_id_to_index[song_id]
            sim_scores += similarity_matrix[idx]

    # Exclude already listened songs
    listened_indices = [song_id_to_index[sid] for sid in user_history if sid in song_id_to_index]
    sim_scores[listened_indices] = -1

    # BOOST songs that match weather genres
    for genre in weather_genres:
        genre_col = f"genre_{genre}"
        if genre_col in song_features.columns:
            sim_scores += song_features[genre_col].values * 0.5

    # Get top 5 recommendations
    top_indices = sim_scores.argsort()[::-1][:5]
    recommended_song_ids = [index_to_song_id[idx] for idx in top_indices]

    return jsonify(recommended_song_ids)

@app.route("/recommendations_moodenhanced", methods=["GET"])
def recommend_moodenhanced():
    user_id = request.args.get("user_id")
    city = request.args.get("city")

    # Detect Mood
    weather_data = detect_mood_from_weather(city) if city else None

    if not weather_data:
        return jsonify({"error": "Unable to detect weather/mood for city"}), 400

    # Debug print
    print("🎯 Debug Weather Info:")
    print("City:", city)
    print("Detected Mood:", weather_data["mood"] if weather_data else None)
    print("Weather Condition:", weather_data["weather"] if weather_data else None)
    print("Time of City:", weather_data["time_of_day"] if weather_data else None)
    

    # Get top 5 recommendations
    recommended_songs = recommend_songs_for_user(
        user_id=user_id,
        weather=weather_data["weather"],
        timeOfDay=weather_data["time_of_day"],
        top_n=5
    )

    return jsonify(recommended_songs)

@app.route("/recommendations_moodenhanced_deeplearning", methods=["GET"])
def recommend_moodenhanced_deeplearning():
    user_id = request.args.get("user_id")
    city = request.args.get("city")

    # Detect Mood
    weather_data = detect_mood_from_weather(city) if city else None

    if not weather_data:
        return jsonify({"error": "Unable to detect weather/mood for city"}), 400

    # Debug print
    print("🎯 Debug Weather Info:")
    print("City:", city)
    print("Detected Mood:", weather_data["mood"] if weather_data else None)
    print("Weather Condition:", weather_data["weather"] if weather_data else None)
    print("Time of City:", weather_data["time_of_day"] if weather_data else None)
    

    # Get top 5 recommendations
    recommended_songs = recommend_songs_for_user_enhanced(
        user_id=user_id,
        weather=weather_data["weather"],
        timeOfDay=weather_data["time_of_day"],
        top_n=5
    )

    return jsonify(recommended_songs)

@app.route("/recommendations_moodenhanced_hybrid", methods=["GET"])
def recommend_moodenhanced_hybrid():
    user_id = request.args.get("user_id")
    city = request.args.get("city")

    # Check user listening history data
    history = pd.read_csv(f"recommendation_personalized_randomforest/listening_history_data/{user_id}_listening_data.csv")
    history_count = len(history)

    MIN_HISTORY = 30

    # Detect Mood
    weather_data = detect_mood_from_weather(city) if city else None

    if not weather_data:
        return jsonify({"error": "Unable to detect weather/mood for city"}), 400

    # Debug print
    print("🎯 Debug Weather Info:")
    print("City:", city)
    print("Detected Mood:", weather_data["mood"] if weather_data else None)
    print("Weather Condition:", weather_data["weather"] if weather_data else None)
    print("Time of City:", weather_data["time_of_day"] if weather_data else None)
    
    # If User listening history data has minimum of MIN_HISTORY use DL model
    # Otherwise use ML model
    if history_count >= MIN_HISTORY:
        try:
            print("Using DEEP LEARNING model")

            # Get top 5 recommendations using DL
            recommended_songs = recommend_songs_for_user_enhanced(
                user_id=user_id,
                weather=weather_data["weather"],
                timeOfDay=weather_data["time_of_day"],
                top_n=5
            )
        
        except Exception as e:
            # If DL fails, use ML model
            print("DL failed — fallback to ML:", str(e))

            recommended_songs = recommend_songs_for_user_ml(
                user_id=user_id,
                weather=weather_data["weather"],
                timeOfDay=weather_data["time_of_day"],
                top_n=5
            )
    else:
        recommended_songs = recommend_songs_for_user_ml(
            user_id=user_id,
            weather=weather_data["weather"],
            timeOfDay=weather_data["time_of_day"],
            top_n=5
        )

    return jsonify(recommended_songs)

@app.route("/recommendations_groupplaylist", methods=["GET"])
def recommend_groupplaylist():
    group_id = request.args.get("group_id") 

    # Get top 10 recommendations
    recommended_songs = recommend_songs_for_group_ml(
        group_id=group_id,
    )

    return jsonify(recommended_songs.to_dict(orient="records"))

@app.route("/recommendations_groupplaylist_deeplearning", methods=["GET"])
def recommend_groupplaylist_deeplearning():
    group_id = request.args.get("group_id") 

    # Get top 10 recommendations
    recommended_songs = group_recommendation_DL(
        group_id=group_id,
    )

    return jsonify(recommended_songs.to_dict(orient="records"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
