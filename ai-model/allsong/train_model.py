import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load processed data
df = pd.read_csv("processed_data.csv")
df = df[["_id", "genre", "language", "type"]]

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
onehot_features = encoder.fit_transform(df[["genre","language","type"]])

# Attach one-hot features to song_id
onehot_df = pd.DataFrame(onehot_features, columns=encoder.get_feature_names_out())
onehot_df["_id"] = df["_id"].values

# Aggregate by song_id (if a song appears multiple times)
song_features = onehot_df.groupby("_id").mean().reset_index()

# Keep song_id separately
song_ids = song_features["_id"].values

# Feature matrix (only numeric features)
song_features_matrix = song_features.drop(columns=["_id"]).values

# Compute cosine similarity between songs
similarity_matrix = cosine_similarity(song_features_matrix)

# Map song_id to index
song_id_to_index = {song_id: idx for idx, song_id in enumerate(song_features["_id"].values)}
index_to_song_id = {idx: song_id for song_id, idx in song_id_to_index.items()}

# Save everything needed for recommendations
joblib.dump({
    "similarity_matrix": similarity_matrix,
    "song_id_to_index": song_id_to_index,
    "index_to_song_id": index_to_song_id,
}, "trained_model.pkl")

print("✅ Model trained and saved as trained_model.pkl")

def recommend_songs(user_history_song_ids, top_n=5):
    """
    Given a list of song_ids the user listened to, recommend top_n similar songs
    """
    import numpy as np
    
    # Aggregate similarity scores
    sim_scores = np.zeros(len(song_features))
    for song_id in user_history_song_ids:
        if song_id in song_id_to_index:
            idx = song_id_to_index[song_id]
            sim_scores += similarity_matrix[idx]
    
    # Avoid recommending songs already listened
    listened_indices = [song_id_to_index[sid] for sid in user_history_song_ids if sid in song_id_to_index]
    sim_scores[listened_indices] = -1  # exclude already listened
    
    # Get top N indices
    top_indices = sim_scores.argsort()[::-1][:top_n]
    recommended_song_ids = [index_to_song_id[idx] for idx in top_indices]
    
    return recommended_song_ids

plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    xticklabels=song_features["_id"],
    yticklabels=song_features["_id"],
    cmap="viridis"
)
plt.title("Song Cosine Similarity Heatmap")
plt.xlabel("Songs")
plt.ylabel("Songs")
plt.show()
