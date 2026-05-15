"""Minimal local web server for the Magic Movie demo app."""

from __future__ import annotations

import argparse
import json
import mimetypes
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from .data import GENRES, load_dataset
from .features import build_model_frame
from .models import coefficient_intervals, fit_model_suite
from .recommender import Preferences, contribution_table, recommend

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
HF_POSTERS_DIR = ROOT / "data" / "hf_poster_images"


@lru_cache(maxsize=1)
def app_state() -> dict[str, object]:
    dataset = load_dataset()
    frame = build_model_frame(dataset.movies, dataset.ratings)
    models = fit_model_suite(frame)
    coefs = coefficient_intervals(frame)
    return {
        "movies": dataset.movies,
        "ratings": dataset.ratings,
        "frame": frame,
        "models": models,
        "coefs": coefs,
    }


class MagicMovieHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path, parse_qs(parsed.query))
            return
        self._serve_static(parsed.path)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_api(self, path: str, params: dict[str, list[str]]) -> None:
        state = app_state()
        if path == "/api/bootstrap":
            self._json(
                {
                    "genres": GENRES,
                    "users": sorted(state["frame"]["user_id"].unique().tolist()),
                    "ratingsCount": int(len(state["ratings"])),
                    "moviesCount": int(len(state["movies"])),
                }
            )
            return

        if path == "/api/recommend":
            prefs = _preferences_from_params(params)
            recs = recommend(state["frame"], state["movies"], prefs, n=8)
            first = recs.iloc[0]
            self._json(
                {
                    "recommendations": _records(recs),
                    "contributions": contribution_table(first, prefs),
                }
            )
            return

        if path == "/api/models":
            self._json(
                {
                    "models": [
                        {
                            "name": model.name,
                            "rmse": model.rmse,
                            "mae": model.mae,
                            "mse": model.mse,
                            "aic": model.aic,
                            "bic": model.bic,
                        }
                        for model in state["models"].values()
                    ],
                    "coefficients": _records(state["coefs"].head(12)),
                }
            )
            return

        if path == "/api/diagnostics":
            frame = state["frame"]
            residuals = frame["adjusted_rating"].to_numpy()
            self._json(
                {
                    "ratingHistogram": _histogram(frame["rating"].to_numpy(), bins=10),
                    "residualHistogram": _histogram(residuals, bins=14),
                    "assumptions": [
                        "Ratings from the same user are correlated.",
                        "Ratings for the same movie are correlated.",
                        "Naive OLS intervals treat rows as independent, so they can be too narrow.",
                        "Cluster bootstrap by user communicates this limitation directly.",
                    ],
                }
            )
            return

        self._json({"error": "Not found"}, status=404)

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            path = "/index.html"
        if path.startswith("/hf_posters/"):
            file_path = (HF_POSTERS_DIR / path.removeprefix("/hf_posters/")).resolve()
            allowed_dir = HF_POSTERS_DIR.resolve()
        else:
            file_path = (WEB_DIR / path.lstrip("/")).resolve()
            allowed_dir = WEB_DIR.resolve()
        if not str(file_path).startswith(str(allowed_dir)) or not file_path.exists():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Magic Movie local web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MagicMovieHandler)
    print(f"Magic Movie running at http://{args.host}:{args.port}")
    server.serve_forever()


def _preferences_from_params(params: dict[str, list[str]]) -> Preferences:
    genres = tuple(_split(params.get("genres", ["Comedy,Drama"])[0]))
    return Preferences(
        user_id=int(params.get("user", ["1"])[0]),
        genres=genres,
        mood=params.get("mood", ["thoughtful"])[0],
        confidence_preference=float(params.get("confidence", ["0.5"])[0]),
        mainstream_vs_hidden_gem=float(params.get("hiddenGem", ["0.55"])[0]),
        adventurous_vs_safe=float(params.get("adventure", ["0.45"])[0]),
        complexity_preference=float(params.get("effort", ["0.5"])[0]),
        mode=params.get("mode", ["balanced"])[0],
    )


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _histogram(values: np.ndarray, bins: int) -> list[dict[str, float]]:
    counts, edges = np.histogram(values, bins=bins)
    return [
        {"x": float((edges[i] + edges[i + 1]) / 2), "y": int(counts[i])}
        for i in range(len(counts))
    ]


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()
