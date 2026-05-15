from magicmovie.data import make_demo_dataset
from magicmovie.features import build_model_frame, feature_columns, movie_candidate_frame
from magicmovie.models import coefficient_intervals, fit_model_suite
from magicmovie.recommender import Preferences, contribution_table, recommend


def test_feature_frame_contains_adjusted_target_and_features():
    dataset = make_demo_dataset()
    frame = build_model_frame(dataset.movies, dataset.ratings)
    assert "adjusted_rating" in frame.columns
    for col in feature_columns():
        assert col in frame.columns


def test_model_suite_returns_comparable_metrics():
    dataset = make_demo_dataset()
    frame = build_model_frame(dataset.movies, dataset.ratings)
    models = fit_model_suite(frame)
    assert set(models.keys()) >= {"OLS", "Ridge", "LASSO", "Logistic"}
    for model in models.values():
        assert model.rmse > 0
        assert model.mae > 0


def test_recommendations_include_intervals_and_explanations():
    dataset = make_demo_dataset()
    frame = build_model_frame(dataset.movies, dataset.ratings)
    prefs = Preferences(user_id=1, genres=("Comedy", "Drama"))
    recs = recommend(frame, dataset.movies, prefs, n=5)
    assert len(recs) == 5
    assert "bootstrap_low_rating" in recs.columns
    assert "bootstrap_high_rating" in recs.columns
    assert "explanation" in recs.columns
    first = recs.iloc[0]
    table = contribution_table(first, prefs)
    assert len(table) > 0


def test_candidate_frame_excludes_already_rated_movies():
    dataset = make_demo_dataset()
    frame = build_model_frame(dataset.movies, dataset.ratings)
    prefs = Preferences(user_id=1, genres=("Comedy", "Drama"))
    candidates = movie_candidate_frame(frame, dataset.movies, prefs.user_id)
    rated = set(frame.loc[frame["user_id"] == prefs.user_id, "movie_id"])
    assert candidates["movie_id"].isin(rated).sum() == 0


def test_coefficient_intervals_have_expected_columns():
    dataset = make_demo_dataset()
    frame = build_model_frame(dataset.movies, dataset.ratings)
    coefs = coefficient_intervals(frame)
    for col in ("feature", "estimate", "p_value"):
        assert col in coefs.columns
