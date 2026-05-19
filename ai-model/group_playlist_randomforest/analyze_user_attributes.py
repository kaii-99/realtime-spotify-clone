import argparse
from pathlib import Path

import pandas as pd


FEATURES = ["genre", "language", "type"]


def _value_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna("UNKNOWN").astype(str).value_counts()


def _print_feature_counts(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for feature in FEATURES:
        counts = _value_counts(df, feature)
        print(f"\n{feature.capitalize()} Count")
        for label, cnt in counts.items():
            print(f"  {label}: {cnt}")


def _print_aggregate(df: pd.DataFrame) -> None:
    print("\nAggregate Summary")
    print("-----------------")
    total_rows = len(df)
    print(f"Total Rows: {total_rows}")

    for feature in FEATURES:
        counts = _value_counts(df, feature)
        print(f"\n{feature.capitalize()} Aggregate")
        for label, cnt in counts.items():
            pct = (cnt / total_rows * 100.0) if total_rows else 0.0
            print(f"  {label}: {cnt} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Genre/Language/Type counts for selected users."
    )
    parser.add_argument(
        "--user_ids",
        nargs="+",
        help="One or more user IDs, e.g. --user_ids 690f... 690f...",
    )
    parser.add_argument(
        "--all_users",
        action="store_true",
        help="Analyze all users found in listening_history_data.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    history_dir = base_dir / "listening_history_data"

    if not history_dir.exists():
        raise FileNotFoundError(f"Missing directory: {history_dir}")

    if args.all_users:
        user_ids = sorted(
            p.name.replace("_listening_data.csv", "")
            for p in history_dir.glob("*_listening_data.csv")
        )
    else:
        user_ids = args.user_ids or []

    if not user_ids:
        raise ValueError("Provide --user_ids ... or use --all_users")

    frames = []
    for user_id in user_ids:
        csv_path = history_dir / f"{user_id}_listening_data.csv"
        if not csv_path.exists():
            print(f"\nSkip {user_id}: file not found -> {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            print(f"\nSkip {user_id}: missing columns {missing}")
            continue

        print(f"\nUser: {user_id} | Rows: {len(df)}")
        _print_feature_counts("User Attribute Counts", df)
        frames.append(df.assign(user_id=user_id))

    if not frames:
        raise ValueError("No valid user CSVs found to analyze.")

    merged = pd.concat(frames, ignore_index=True)
    _print_aggregate(merged)


if __name__ == "__main__":
    main()

