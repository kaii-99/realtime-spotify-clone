from flask import Flask, request, jsonify
import joblib
import numpy as np
from weather_detection.weather_mood import detect_mood_from_weather

app = Flask(__name__)

# Load the trained model data
model_data = joblib.load("trained_model.pkl")
similarity_matrix = model_data["similarity_matrix"]
song_id_to_index = model_data["song_id_to_index"]
index_to_song_id = model_data["index_to_song_id"]

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
    if not song_ids_param:
        return jsonify({"error": "songIds query parameter is required"}), 400

    user_history = song_ids_param.split(",")

    # Initialize similarity scores
    sim_scores = np.zeros(len(song_id_to_index))

    for song_id in user_history:
        if song_id in song_id_to_index:
            idx = song_id_to_index[song_id]
            sim_scores += similarity_matrix[idx]

    # Exclude already listened songs
    listened_indices = [song_id_to_index[sid] for sid in user_history if sid in song_id_to_index]
    sim_scores[listened_indices] = -1

    # Get top 5 recommendations
    top_indices = sim_scores.argsort()[::-1][:5]
    recommended_song_ids = [index_to_song_id[idx] for idx in top_indices]

    return jsonify(recommended_song_ids)

if __name__ == "__main__":
    app.run(port=4000)
