import argparse
import json
import os
import pandas as pd
import torch

from deep_learning_model_pytorch_SHAP import (
    load_user_data,
    train_model,
    recommend_songs,
    RecommenderNN,
)


def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run personalized deep-learning SHAP for a custom weather/time context."
    )
    parser.add_argument("--user_id", required=True, help="User ID to load processed data/model context")
    parser.add_argument("--weather", required=True, help="Weather label, e.g. Clouds/Rain")
    parser.add_argument("--timeOfDay", required=True, help="Time label, e.g. day/night")
    parser.add_argument("--top_n", type=int, default=5, help="Number of songs to show")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--explain", action="store_true", help="Enable SHAP output")
    parser.add_argument(
        "--context_rerank",
        action="store_true",
        help="Apply light context-fit rerank by user weather/time history",
    )
    parser.add_argument(
        "--exclude_listened",
        action="store_true",
        help="Exclude songs already in the user's listening history (server-like)",
    )
    parser.add_argument(
        "--model_mode",
        choices=["saved", "train"],
        default="saved",
        help="Use saved production model (server-like) or train a fresh model",
    )
    args = parser.parse_args()

    df, weather_enc, time_enc, _, _, _, _ = load_user_data(args.user_id)

    all_songs = pd.read_csv("all_songs_encoded.csv")
    all_songs.columns = all_songs.columns.str.strip()

    if args.model_mode == "saved":
        model_path = os.path.join("model", args.user_id, f"{args.user_id}_personalized_model_pytorch.pth")
        song_map_path = os.path.join("model", args.user_id, f"{args.user_id}_song_mapping.json")

        with open(song_map_path, "r", encoding="utf-8") as f:
            idx_to_song_encoded = json.load(f)
        song_encoded_to_idx = {int(v): int(k) for k, v in idx_to_song_encoded.items()}
        num_songs = len(idx_to_song_encoded)

        model = RecommenderNN(input_dim=5, hidden_dim=64, output_dim=num_songs)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        num_songs = len(df["song_idx"].unique())
        model = train_model(df, num_songs=num_songs, epochs=args.epochs)
        song_encoded_to_idx = {song: idx for idx, song in enumerate(sorted(df["song_id_encoded"].unique()))}

    weather_val = safe_transform(weather_enc, args.weather)
    time_val = safe_transform(time_enc, args.timeOfDay)

    top_songs = recommend_songs(
        model=model,
        all_songs_df=all_songs,
        weather_enc_val=weather_val,
        time_enc_val=time_val,
        song_encoded_to_idx=song_encoded_to_idx,
        user_id=args.user_id,
        exclude_listened=args.exclude_listened,
        context_rerank=args.context_rerank,
        top_n=args.top_n,
        explain=args.explain,
    )

    print(f"\nTop songs for Weather: {args.weather}, Time: {args.timeOfDay}")
    print(top_songs[["song_id", "score"]])

    if args.explain and "shap_values" in top_songs.columns:
        feature_names = [
            "genre_encoded",
            "language_encoded",
            "type_encoded",
            "weather_encoded",
            "timeOfDay_encoded",
        ]
        for _, song_row in top_songs.iterrows():
            print(f"\nSong {song_row['song_id']}:")
            for name, val in zip(feature_names, song_row["shap_values"]):
                print(f"  {name}: {val:.3f}")


if __name__ == "__main__":
    main()
