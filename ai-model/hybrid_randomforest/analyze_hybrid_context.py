import argparse
from pathlib import Path

import pandas as pd


FEATURES = ["genre", "language", "type"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["weather", "timeOfDay", *FEATURES]:
        if col in out.columns:
            out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


def _collect_user_frames(history_dir: Path, user_ids: list[str]) -> pd.DataFrame:
    frames = []
    for user_id in user_ids:
        csv_path = history_dir / f"{user_id}_listening_data.csv"
        if not csv_path.exists():
            print(f"Skip {user_id}: file not found -> {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        missing = [c for c in ["weather", "timeOfDay", *FEATURES] if c not in df.columns]
        if missing:
            print(f"Skip {user_id}: missing columns {missing}")
            continue

        df = _clean(df)
        df["user_id"] = user_id
        frames.append(df)

    if not frames:
        raise ValueError("No valid listening history files found.")
    return pd.concat(frames, ignore_index=True)


def _print_context_distribution(df: pd.DataFrame, top_attr_n: int) -> None:
    overall = {f: df[f].value_counts(normalize=True) for f in FEATURES}
    contexts = (
        df.groupby(["weather", "timeOfDay"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("\nContext Counts")
    print("--------------")
    for _, row in contexts.iterrows():
        print(f"{row['weather']} | {row['timeOfDay']} -> {int(row['count'])}")

    for _, ctx in contexts.iterrows():
        weather = ctx["weather"]
        time_of_day = ctx["timeOfDay"]
        ctx_df = df[(df["weather"] == weather) & (df["timeOfDay"] == time_of_day)]
        total = len(ctx_df)
        print(f"\nContext: weather={weather}, timeOfDay={time_of_day} (rows={total})")
        print("-" * 60)

        for feature in FEATURES:
            counts = ctx_df[feature].value_counts()
            print(f"{feature.capitalize()} Top {top_attr_n}")
            for label, cnt in counts.head(top_attr_n).items():
                pct = cnt / total * 100.0
                overall_pct = overall[feature].get(label, 0.0)
                lift = (cnt / total) / overall_pct if overall_pct > 0 else 0.0
                print(f"  {label}: {cnt} ({pct:.1f}%), lift={lift:.2f}")


def _evaluate_recommendations(
    df: pd.DataFrame,
    all_songs_path: Path,
    weather: str,
    time_of_day: str,
    recommended_song_ids: list[str],
    top_attr_n: int,
) -> None:
    all_songs = pd.read_csv(all_songs_path)
    all_songs.columns = all_songs.columns.str.strip()
    required_cols = ["song_id", *FEATURES]
    missing = [c for c in required_cols if c not in all_songs.columns]
    if missing:
        raise ValueError(f"all_songs file missing columns: {missing}")

    ctx_df = df[(df["weather"] == weather) & (df["timeOfDay"] == time_of_day)]
    if ctx_df.empty:
        raise ValueError(f"No listening rows for context weather={weather}, timeOfDay={time_of_day}")

    top_sets = {
        f: set(ctx_df[f].value_counts().head(top_attr_n).index.tolist())
        for f in FEATURES
    }

    rec_df = all_songs[all_songs["song_id"].astype(str).isin(recommended_song_ids)].copy()
    rec_df["song_id"] = rec_df["song_id"].astype(str)
    rec_df["match_genre"] = rec_df["genre"].astype(str).isin(top_sets["genre"])
    rec_df["match_language"] = rec_df["language"].astype(str).isin(top_sets["language"])
    rec_df["match_type"] = rec_df["type"].astype(str).isin(top_sets["type"])
    rec_df["match_all_3"] = rec_df["match_genre"] & rec_df["match_language"] & rec_df["match_type"]

    k = len(rec_df)
    if k == 0:
        print("\nNo recommended song IDs matched rows in all_songs file.")
        return

    print(f"\nRecommendation Context Match @K (K={k}) for weather={weather}, timeOfDay={time_of_day}")
    print("-----------------------------------------------------------------------")
    print(f"Genre Match Rate: {rec_df['match_genre'].mean() * 100:.1f}%")
    print(f"Language Match Rate: {rec_df['match_language'].mean() * 100:.1f}%")
    print(f"Type Match Rate: {rec_df['match_type'].mean() * 100:.1f}%")
    print(f"All-3 Match Rate: {rec_df['match_all_3'].mean() * 100:.1f}%")

    print("\nPer-song match details")
    cols = ["song_id", "genre", "language", "type", "match_genre", "match_language", "match_type", "match_all_3"]
    print(rec_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid context analysis: distribution + lift by weather/timeOfDay."
    )
    parser.add_argument("--user_ids", nargs="+", help="Selected user IDs")
    parser.add_argument("--all_users", action="store_true", help="Analyze all users in listening_history_data")
    parser.add_argument("--top_attr_n", type=int, default=3, help="Top-N attributes to print per feature")
    parser.add_argument("--weather", help="Optional context weather for recommendation evaluation")
    parser.add_argument("--timeOfDay", help="Optional context timeOfDay for recommendation evaluation")
    parser.add_argument("--recommended_song_ids", nargs="+", help="Optional recommended song IDs to evaluate")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    history_dir = base_dir / "listening_history_data"
    all_songs_path = base_dir / "all_songs_encoded.csv"

    if args.all_users:
        user_ids = sorted(
            p.name.replace("_listening_data.csv", "")
            for p in history_dir.glob("*_listening_data.csv")
        )
    else:
        user_ids = args.user_ids or []

    if not user_ids:
        raise ValueError("Provide --user_ids ... or use --all_users")

    df = _collect_user_frames(history_dir, user_ids)
    print(f"Analyzing users: {', '.join(user_ids)}")
    print(f"Total rows: {len(df)}")
    _print_context_distribution(df, top_attr_n=args.top_attr_n)

    eval_requested = args.weather and args.timeOfDay and args.recommended_song_ids
    if eval_requested:
        _evaluate_recommendations(
            df=df,
            all_songs_path=all_songs_path,
            weather=args.weather,
            time_of_day=args.timeOfDay,
            recommended_song_ids=[str(x) for x in args.recommended_song_ids],
            top_attr_n=args.top_attr_n,
        )


if __name__ == "__main__":
    main()

