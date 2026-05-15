
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Documentary",
]

MOOD_TO_GENRES = {
    "fun": ["Comedy", "Adventure"],
    "thoughtful": ["Drama", "Documentary"],
    "intense": ["Thriller", "Action"],
    "warm": ["Romance", "Comedy", "Drama"],
    "curious": ["Sci-Fi", "Documentary"],
}

DIRECTORS = [
    "Ava DuVernay",
    "Michel Gondry",
    "Sofia Coppola",
    "Bong Joon-ho",
    "Greta Gerwig",
    "Barry Jenkins",
    "Mira Nair",
    "Denis Villeneuve",
    "Chloe Zhao",
    "Rian Johnson",
]


@dataclass(frozen=True)
class DemoDataset:
    movies: pd.DataFrame
    ratings: pd.DataFrame


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-latest-small"
METADATA_CACHE = Path(__file__).resolve().parents[1] / "data" / "movie_metadata_cache.json"
HF_POSTERS_DIR = Path(__file__).resolve().parents[1] / "data" / "hf_poster_images"


def load_dataset(random_state: int = 7) -> DemoDataset:
    """Load MovieLens Latest Small when available, otherwise use demo data."""

    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    links_path = DATA_DIR / "links.csv"
    if ratings_path.exists() and movies_path.exists():
        return load_movielens_dataset(ratings_path, movies_path, links_path if links_path.exists() else None)
    return make_demo_dataset(random_state=random_state)


def load_movielens_dataset(
    ratings_path: Path,
    movies_path: Path,
    links_path: Path | None = None,
) -> DemoDataset:
    """Load MovieLens CSV files and adapt them to Magic Movie's feature schema."""

    ratings = pd.read_csv(ratings_path)[["userId", "movieId", "rating", "timestamp"]].rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )
    movies_raw = pd.read_csv(movies_path)
    movies = movies_raw.rename(columns={"movieId": "movie_id"}).copy()
    title_year = movies["title"].apply(_split_title_year)
    movies["title"] = title_year.apply(lambda value: _normalize_sort_title(value[0]))
    movies["year"] = title_year.apply(lambda value: value[1])

    genre_sets = movies["genres"].fillna("").str.split("|")
    for genre in GENRES:
        movies[genre] = genre_sets.apply(lambda values: int(genre in values))
    movies["rating_count"] = movies["movie_id"].map(ratings.groupby("movie_id").size()).fillna(0).astype(int)
    movies["popularity_score"] = np.log1p(movies["rating_count"])
    movie_mean = ratings.groupby("movie_id")["rating"].mean()
    movie_var = ratings.groupby("movie_id")["rating"].var()
    global_mean = ratings["rating"].mean()
    movies["imdb_rating"] = movies["movie_id"].map(movie_mean).fillna(global_mean).mul(2).round(1).clip(1, 10)
    movies["hidden_gem"] = movies["rating_count"].rank(pct=True, ascending=True).round(3)
    movies["runtime"] = 82 + (movies["movie_id"] % 74)
    movies["director"] = "Metadata unavailable"
    movies["poster_url"] = None

    if links_path and links_path.exists():
        links = pd.read_csv(links_path).rename(columns={"movieId": "movie_id"})
        movies = movies.merge(links, on="movie_id", how="left")
        movies = _enrich_movie_metadata(movies)
    else:
        movies["imdbId"] = pd.NA
        movies["tmdbId"] = pd.NA

    return DemoDataset(movies=movies.fillna(0), ratings=ratings)


def make_demo_dataset(
    n_users: int = 180,
    n_movies: int = 85,
    random_state: int = 7,
) -> DemoDataset:
    """Generate correlated ratings with user/movie structure.
    
    """

    rng = np.random.default_rng(random_state)
    movies = _make_movies(n_movies, rng)

    global_mean = 3.45
    user_bias = rng.normal(0, 0.45, n_users)
    user_genre_weights = rng.normal(0, 0.34, (n_users, len(GENRES)))
    user_activity = rng.poisson(20, n_users) + 8

    rows = []
    genre_matrix = movies[GENRES].to_numpy()
    movie_quality = movies["quality"].to_numpy()
    popularity_bias = movies["popularity_score"].to_numpy() * 0.18

    for user_idx in range(n_users):
        max_watched = max(1, n_movies - max(5, n_movies // 6))
        watched = rng.choice(
            n_movies,
            size=min(max_watched, user_activity[user_idx]),
            replace=False,
            p=_softmax(movie_quality + popularity_bias),
        )
        for movie_idx in watched:
            preference = genre_matrix[movie_idx] @ user_genre_weights[user_idx]
            signal = (
                global_mean
                + user_bias[user_idx]
                + movie_quality[movie_idx]
                + 0.24 * preference
                + 0.10 * movies.iloc[movie_idx]["hidden_gem"]
            )
            rating = np.clip(signal + rng.normal(0, 0.55), 0.5, 5.0)
            rows.append(
                {
                    "user_id": user_idx + 1,
                    "movie_id": movie_idx + 1,
                    "rating": round(float(rating * 2) / 2, 1),
                    "timestamp": 1_700_000_000 + len(rows) * 971,
                }
            )

    ratings = pd.DataFrame(rows)
    return DemoDataset(movies=movies.drop(columns=["quality"]), ratings=ratings)


def _split_title_year(title: str) -> tuple[str, int]:
    match = re.search(r"\((\d{4})\)\s*$", title)
    if not match:
        return title, 0
    return title[: match.start()].strip(), int(match.group(1))


def _normalize_sort_title(title: str) -> str:
    match = re.match(r"^(?P<base>.+),\s+(?P<article>The|A|An)$", title)
    if not match:
        return title
    return f"{match.group('article')} {match.group('base')}"


def _enrich_movie_metadata(movies: pd.DataFrame) -> pd.DataFrame:
    cache = _read_metadata_cache()
    omdb_key = os.getenv("OMDB_API_KEY")
    tmdb_key = os.getenv("TMDB_API_KEY")
    changed = False

    for index, row in movies.iterrows():
        imdb_id = _format_imdb_id(row.get("imdbId"))
        if not imdb_id:
            continue
        metadata = cache.get(imdb_id)
        can_fetch_metadata = bool(omdb_key or tmdb_key)
        needs_retry = metadata is not None and can_fetch_metadata and not metadata.get("poster_url")
        if metadata is None or needs_retry:
            fetched = _fetch_omdb_metadata(imdb_id, omdb_key) or _fetch_tmdb_metadata(imdb_id, tmdb_key) or {}
            metadata = {**(metadata or {}), **fetched}
            cache[imdb_id] = metadata
            changed = True
        local_poster_url = _local_hf_poster_url(row)
        if local_poster_url and not metadata.get("poster_url"):
            metadata["poster_url"] = local_poster_url
            cache[imdb_id] = metadata
            changed = True
        if metadata.get("director"):
            movies.at[index, "director"] = metadata["director"]
        if metadata.get("runtime"):
            movies.at[index, "runtime"] = metadata["runtime"]
        if metadata.get("imdb_rating"):
            movies.at[index, "imdb_rating"] = metadata["imdb_rating"]
        if metadata.get("poster_url"):
            movies.at[index, "poster_url"] = metadata["poster_url"]

    if changed:
        _write_metadata_cache(cache)
    return movies


def _local_hf_poster_url(row: pd.Series) -> str | None:
    tmdb_id = row.get("tmdbId")
    if not HF_POSTERS_DIR.exists() or pd.isna(tmdb_id):
        return None
    try:
        poster_id = str(int(float(tmdb_id)))
    except (TypeError, ValueError):
        return None
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        if (HF_POSTERS_DIR / f"{poster_id}{suffix}").exists():
            return f"/hf_posters/{poster_id}{suffix}"
    return None


def _fetch_omdb_metadata(imdb_id: str, api_key: str | None) -> dict[str, object] | None:
    if not api_key:
        return None
    params = urllib.parse.urlencode({"i": imdb_id, "apikey": api_key})
    try:
        with urllib.request.urlopen(f"https://www.omdbapi.com/?{params}", timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError:
        return None
    if payload.get("Response") != "True":
        return None
    return {
        "director": payload.get("Director") or None,
        "runtime": _runtime_minutes(payload.get("Runtime")),
        "imdb_rating": _float_or_none(payload.get("imdbRating")),
        "poster_url": None if payload.get("Poster") in {None, "N/A"} else payload.get("Poster"),
    }


def _fetch_tmdb_metadata(imdb_id: str, api_key: str | None) -> dict[str, object] | None:
    if not api_key:
        return None
    params = urllib.parse.urlencode({"api_key": api_key, "external_source": "imdb_id"})
    try:
        with urllib.request.urlopen(f"https://api.themoviedb.org/3/find/{imdb_id}?{params}", timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError:
        return None
    results = payload.get("movie_results") or []
    if not results:
        return None
    movie = results[0]
    poster_path = movie.get("poster_path")
    return {
        "imdb_rating": _float_or_none(movie.get("vote_average")),
        "poster_url": f"https://image.tmdb.org/t/p/w780{poster_path}" if poster_path else None,
    }


def _read_metadata_cache() -> dict[str, dict[str, object]]:
    if not METADATA_CACHE.exists():
        return {}
    try:
        return json.loads(METADATA_CACHE.read_text())
    except json.JSONDecodeError:
        return {}


def _write_metadata_cache(cache: dict[str, dict[str, object]]) -> None:
    METADATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _format_imdb_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        return f"tt{int(value):07d}"
    except (TypeError, ValueError):
        return None


def _runtime_minutes(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def _float_or_none(value: object) -> float | None:
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _make_movies(n_movies: int, rng: np.random.Generator) -> pd.DataFrame:
    title_words = [
        "Aurora",
        "Midnight",
        "Signal",
        "Harbor",
        "Orbit",
        "Lantern",
        "Paper",
        "Velvet",
        "Neon",
        "Summit",
        "Afterglow",
        "Blueprint",
    ]
    title_suffixes = [
        "Road",
        "Theory",
        "Club",
        "Weekend",
        "Protocol",
        "Season",
        "Archive",
        "Letters",
        "Machine",
        "Garden",
    ]

    genre_rows = []
    titles = []
    years = []
    quality = rng.normal(0, 0.42, n_movies)
    popularity = rng.gamma(shape=2.2, scale=80, size=n_movies).astype(int) + 15

    for idx in range(n_movies):
        primary = rng.integers(0, len(GENRES))
        secondary = rng.integers(0, len(GENRES))
        row = np.zeros(len(GENRES), dtype=int)
        row[primary] = 1
        if secondary != primary and rng.random() < 0.55:
            row[secondary] = 1
        genre_rows.append(row)
        titles.append(f"{title_words[idx % len(title_words)]} {title_suffixes[idx % len(title_suffixes)]}")
        years.append(1985 + int(rng.integers(0, 40)))

    movies = pd.DataFrame(genre_rows, columns=GENRES)
    movies.insert(0, "movie_id", np.arange(1, n_movies + 1))
    movies.insert(1, "title", titles)
    movies.insert(2, "year", years)
    movies["director"] = [DIRECTORS[idx % len(DIRECTORS)] for idx in range(n_movies)]
    movies["runtime"] = rng.integers(82, 154, n_movies)
    movies["imdb_rating"] = np.clip(6.2 + quality * 1.8 + rng.normal(0, 0.35, n_movies), 5.1, 9.2).round(1)
    movies["poster_url"] = None
    movies["rating_count"] = popularity
    movies["popularity_score"] = np.log1p(popularity)
    movies["quality"] = quality
    movies["hidden_gem"] = (movies["rating_count"].rank(pct=True, ascending=True)).round(3)
    movies["genres"] = movies[GENRES].apply(
        lambda row: "|".join([genre for genre in GENRES if row[genre] == 1]),
        axis=1,
    )
    return movies


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values)
    exp = np.exp(centered)
    return exp / exp.sum()
