
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import GENRES, MOOD_TO_GENRES
from .features import feature_columns, movie_candidate_frame
from .models import naive_prediction_interval

RESIDUAL_EFFECT_WEIGHT = 0.65
GENRE_COMPLEXITY = {
    "Comedy": 0.25,
    "Romance": 0.35,
    "Animation": 0.30,
    "Adventure": 0.48,
    "Action": 0.52,
    "Drama": 0.62,
    "Documentary": 0.70,
    "Sci-Fi": 0.74,
    "Thriller": 0.78,
    "Crime": 0.72,
}


@dataclass(frozen=True)
class Preferences:
    user_id: int = 1
    genres: tuple[str, ...] = ("Comedy", "Drama")
    mood: str = "thoughtful"
    confidence_preference: float = 0.5
    runtime_preference: str = "any"
    mainstream_vs_hidden_gem: float = 0.55
    adventurous_vs_safe: float = 0.45
    complexity_preference: float = 0.5
    mode: str = "balanced"


def recommend(
    model_frame: pd.DataFrame,
    movies: pd.DataFrame,
    preferences: Preferences,
    n: int = 8,
) -> pd.DataFrame:
    candidates = movie_candidate_frame(model_frame, movies, preferences.user_id)
    naive = naive_prediction_interval(model_frame, candidates)
    scored = pd.concat([candidates.reset_index(drop=True), naive.reset_index(drop=True)], axis=1)
    scored = _add_questionnaire_scores(scored, preferences)
    scored["adjustment_pred"] = RESIDUAL_EFFECT_WEIGHT * scored["residual_pred"]
    scored["predicted_rating"] = (
        scored["bayes_movie_mean"] + scored["adjustment_pred"]
    ).clip(0.5, 5.0)
    scored["prediction_se"] = _prediction_se(model_frame, scored)
    scored["naive_low_rating"] = (
        scored["bayes_movie_mean"] + RESIDUAL_EFFECT_WEIGHT * scored["naive_low"]
    ).clip(0.5, 5.0)
    scored["naive_high_rating"] = (
        scored["bayes_movie_mean"] + RESIDUAL_EFFECT_WEIGHT * scored["naive_high"]
    ).clip(0.5, 5.0)
    scored["bootstrap_low_rating"] = (scored["predicted_rating"] - 1.96 * scored["prediction_se"]).clip(0.5, 5.0)
    scored["bootstrap_high_rating"] = (scored["predicted_rating"] + 1.96 * scored["prediction_se"]).clip(0.5, 5.0)
    scored["bootstrap_se"] = scored["prediction_se"]
    scored["decision_score"] = _final_recommendation_score(scored, preferences)
    scored["explanation"] = scored.apply(lambda row: explain(row, preferences), axis=1)
    columns = [
        "movie_id",
        "title",
        "year",
        "director",
        "runtime",
        "imdb_rating",
        "poster_url",
        "genres",
        "predicted_rating",
        "naive_low_rating",
        "naive_high_rating",
        "bootstrap_low_rating",
        "bootstrap_high_rating",
        "bootstrap_se",
        "genre_match_score",
        "mood_match_score",
        "popularity_match",
        "risk_match",
        "complexity_match",
        "uncertainty_proxy",
        "popularity_pct",
        "bayes_movie_mean",
        "adjustment_pred",
        "decision_score",
        "explanation",
    ] + feature_columns()
    return scored.sort_values("decision_score", ascending=False).head(n)[columns]


def contribution_table(row: pd.Series, preferences: Preferences) -> list[dict[str, object]]:
    rows = [
        {"factor": "Bayesian movie mean", "effect": round(float(row["bayes_movie_mean"]), 3)},
        {"factor": "Residual regression adjustment", "effect": round(float(row["adjustment_pred"]), 3)},
        {"factor": "Genre match", "effect": round(float(row["genre_match_score"]) * 0.35, 3)},
        {"factor": "Mood match", "effect": round(float(row["mood_match_score"]) * 0.25, 3)},
        {"factor": "Popularity match", "effect": round(float(row["popularity_match"]) * 0.15, 3)},
        {"factor": "Complexity match", "effect": round(float(row["complexity_match"]) * 0.10, 3)},
        {"factor": "Uncertainty penalty", "effect": round(float(row["bootstrap_se"]) * -preferences.confidence_preference, 3)},
    ]
    return sorted(rows, key=lambda item: abs(float(item["effect"])), reverse=True)[:6]


def explain(row: pd.Series, preferences: Preferences) -> str:
    width = row["bootstrap_high_rating"] - row["bootstrap_low_rating"]
    genre_hits = [genre for genre in preferences.genres if genre in row["genres"]]
    if genre_hits:
        driver = f"alignment with {', '.join(genre_hits)}"
    elif row["hidden_gem"] > 0.65:
        driver = "hidden-gem potential"
    else:
        driver = "baseline movie and user preference structure"

    uncertainty = "wide" if width > 1.0 else "moderate" if width > 0.65 else "tight"
    return (
        f"Driven primarily by {driver}. The prediction interval is {uncertainty}, "
        "using residual error plus movie-level disagreement and sample size uncertainty."
    )


def _add_questionnaire_scores(scored: pd.DataFrame, preferences: Preferences) -> pd.DataFrame:
    selected = [genre for genre in preferences.genres if genre in GENRES]
    mood_genres = [genre for genre in MOOD_TO_GENRES.get(preferences.mood, []) if genre in GENRES]
    scored["genre_match_score"] = scored[selected].sum(axis=1) / len(selected) if selected else 0.0
    scored["mood_match_score"] = scored[mood_genres].sum(axis=1) / len(mood_genres) if mood_genres else 0.0
    scored["popularity_pct"] = _normalize(scored["log_popularity"])
    scored["popularity_match"] = 1 - (preferences.mainstream_vs_hidden_gem - (1 - scored["popularity_pct"])).abs()
    scored["popularity_match"] = scored["popularity_match"].clip(0, 1)
    scored["uncertainty_proxy"] = _uncertainty_proxy(scored)
    scored["risk_match"] = 1 - (preferences.adventurous_vs_safe - scored["uncertainty_proxy"]).abs()
    scored["risk_match"] = scored["risk_match"].clip(0, 1)
    scored["complexity_proxy"] = _complexity_proxy(scored)
    scored["complexity_match"] = 1 - (preferences.complexity_preference - scored["complexity_proxy"]).abs()
    scored["complexity_match"] = scored["complexity_match"].clip(0, 1)
    return scored


def _prediction_se(model_frame: pd.DataFrame, scored: pd.DataFrame) -> pd.Series:
    residual_std = float(model_frame["adjusted_rating"].std(ddof=1))
    return residual_std * (0.18 + 0.32 * scored["uncertainty_proxy"])


def _final_recommendation_score(scored: pd.DataFrame, preferences: Preferences) -> pd.Series:
    uncertainty_weight = preferences.confidence_preference * (1 - preferences.adventurous_vs_safe)
    hidden_weight = preferences.mainstream_vs_hidden_gem
    return (
        scored["predicted_rating"]
        - uncertainty_weight * scored["bootstrap_se"]
        + hidden_weight * (1 - scored["popularity_pct"])
        + 0.35 * scored["genre_match_score"]
        + 0.25 * scored["mood_match_score"]
        + 0.15 * scored["popularity_match"]
        + 0.10 * scored["complexity_match"]
    )


def _uncertainty_proxy(scored: pd.DataFrame) -> pd.Series:
    count_uncertainty = 1 / np.sqrt(scored["observed_rating_count"].clip(lower=1))
    variance_uncertainty = _normalize(scored["movie_rating_variance"])
    return (0.55 * variance_uncertainty + 0.45 * _normalize(count_uncertainty)).clip(0, 1)


def _complexity_proxy(scored: pd.DataFrame) -> pd.Series:
    genre_complexity = pd.Series(0.0, index=scored.index)
    genre_count = pd.Series(0, index=scored.index)
    for genre, value in GENRE_COMPLEXITY.items():
        if genre not in scored:
            continue
        genre_complexity += scored[genre] * value
        genre_count += scored[genre]
    genre_complexity = genre_complexity / genre_count.clip(lower=1)
    return (0.65 * genre_complexity + 0.35 * _normalize(scored["movie_rating_variance"])).clip(0, 1)


def _normalize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(0.0, index=values.index)
    return (values - minimum) / (maximum - minimum)
