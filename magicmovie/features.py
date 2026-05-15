"""Feature engineering for interpretable recommendation models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GENRES

MOVIE_SHRINKAGE_ALPHA = 20


def build_model_frame(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return one row per rating with baseline-adjusted target and features."""

    frame = ratings.merge(movies, on="movie_id", how="left")
    global_mean = frame["rating"].mean()
    user_stats = frame.groupby("user_id")["rating"].agg(
        user_mean="mean",
        user_rating_variance="var",
        user_activity="count",
    )
    movie_stats = frame.groupby("movie_id")["rating"].agg(
        movie_mean="mean",
        movie_rating_variance="var",
        observed_rating_count="count",
    )
    movie_stats["bayes_movie_mean"] = (
        movie_stats["observed_rating_count"]
        / (movie_stats["observed_rating_count"] + MOVIE_SHRINKAGE_ALPHA)
        * movie_stats["movie_mean"]
        + MOVIE_SHRINKAGE_ALPHA
        / (movie_stats["observed_rating_count"] + MOVIE_SHRINKAGE_ALPHA)
        * global_mean
    )

    frame = frame.join(user_stats, on="user_id").join(movie_stats, on="movie_id")
    frame["global_mean"] = global_mean
    frame["adjusted_rating"] = (
        frame["rating"] - frame["user_mean"] - frame["bayes_movie_mean"] + frame["global_mean"]
    )
    frame["liked"] = (frame["rating"] >= 4).astype(int)
    frame["log_popularity"] = np.log1p(frame["rating_count"])
    frame["niche_user"] = (
        frame.groupby("user_id")["rating_count"].transform("mean")
        < frame["rating_count"].median()
    ).astype(int)

    for genre in GENRES:
        user_affinity = pd.Series(
            {
                user_id: _safe_genre_affinity(group, genre)
                for user_id, group in frame.groupby("user_id")
            },
            name=f"{genre}_affinity",
        )
        frame = frame.join(user_affinity, on="user_id")
        frame[f"{genre}_match"] = frame[genre] * frame[f"{genre}_affinity"]

    frame["popularity_preference"] = (
        frame["log_popularity"] - frame.groupby("user_id")["log_popularity"].transform("mean")
    )
    frame["hidden_gem_preference"] = frame["hidden_gem"] * frame["niche_user"]
    return frame.fillna(0)


def feature_columns() -> list[str]:
    """Columns used by the residual regression component."""

    genre_cols = list(GENRES)
    interaction_cols = [f"{genre}_match" for genre in GENRES]
    return (
        genre_cols
        + interaction_cols
        + [
            "log_popularity",
            "hidden_gem",
            "popularity_preference",
            "hidden_gem_preference",
            "niche_user",
        ]
    )


def movie_candidate_frame(
    model_frame: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
) -> pd.DataFrame:
    """Build candidate rows for movies the selected user has not rated."""

    user_rows = model_frame[model_frame["user_id"] == user_id]
    if user_rows.empty:
        raise ValueError(f"Unknown user_id {user_id}")

    rated_ids = set(user_rows["movie_id"])
    candidates = movies[~movies["movie_id"].isin(rated_ids)].copy()
    user_summary = user_rows.iloc[0]
    candidates["user_id"] = user_id
    candidates["user_mean"] = user_summary["user_mean"]
    candidates["user_rating_variance"] = user_summary["user_rating_variance"]
    candidates["user_activity"] = user_summary["user_activity"]
    candidates["global_mean"] = user_summary["global_mean"]
    candidates["movie_mean"] = candidates["movie_id"].map(
        model_frame.groupby("movie_id")["movie_mean"].first()
    ).fillna(user_summary["global_mean"])
    candidates["movie_rating_variance"] = candidates["movie_id"].map(
        model_frame.groupby("movie_id")["movie_rating_variance"].first()
    ).fillna(0)
    candidates["observed_rating_count"] = candidates["movie_id"].map(
        model_frame.groupby("movie_id")["observed_rating_count"].first()
    ).fillna(candidates["rating_count"])
    candidates["bayes_movie_mean"] = candidates["movie_id"].map(
        model_frame.groupby("movie_id")["bayes_movie_mean"].first()
    ).fillna(user_summary["global_mean"])
    candidates["baseline_rating"] = (
        candidates["user_mean"] + candidates["bayes_movie_mean"] - candidates["global_mean"]
    )
    candidates["log_popularity"] = np.log1p(candidates["rating_count"])
    candidates["niche_user"] = int(user_rows["rating_count"].mean() < model_frame["rating_count"].median())

    for genre in GENRES:
        affinity = user_rows[f"{genre}_affinity"].iloc[0]
        candidates[f"{genre}_affinity"] = affinity
        candidates[f"{genre}_match"] = candidates[genre] * affinity

    candidates["popularity_preference"] = (
        candidates["log_popularity"] - user_rows["log_popularity"].mean()
    )
    candidates["hidden_gem_preference"] = candidates["hidden_gem"] * candidates["niche_user"]
    return candidates.fillna(0)


def _safe_genre_affinity(group: pd.DataFrame, genre: str) -> float:
    in_genre = group[group[genre] == 1]
    if in_genre.empty:
        return 0.0
    return float(in_genre["rating"].mean() - group["rating"].mean())
