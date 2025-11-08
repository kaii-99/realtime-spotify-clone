import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # for saving/loading model

# Load preprocessed data
df = pd.read_csv("processed_data.csv")

# Features for content-based recommendation
feature_cols = ["genre_encoded", "language_encoded", "type_encoded"]  # example
X = df[feature_cols]

# Compute similarity matrix
similarity_matrix = cosine_similarity(X)

# Map song_id to index
song_index_map = {song_id: idx for idx, song_id in enumerate(df["song_id"])}

def recommend_songs(user_song_ids, top_n=5):
    scores = {}
    for song_id in user_song_ids:
        if song_id not in song_index_map:
            continue
        idx = song_index_map[song_id]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        for i, score in sim_scores:
            song = df.loc[i, "song_id"]
            scores[song] = scores.get(song, 0) + score
    # Sort by score
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [song_id for song_id, _ in recommended if song_id not in user_song_ids][:top_n]

# Example usage
if __name__ == "__main__":
    test_history = ["6826d6abc88f88d696315f8d", "6826ece55e56db0c7cdc2a76"]
    print(recommend_songs(test_history, top_n=5))
