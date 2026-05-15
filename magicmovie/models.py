"""Statistical estimators and uncertainty intervals for Magic Movie."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns


@dataclass
class ModelResult:
    name: str
    coefficients: pd.Series
    intercept: float
    rmse: float
    mae: float
    mse: float
    aic: float
    bic: float
    residual_std: float


def fit_model_suite(frame: pd.DataFrame, random_state: int = 7) -> dict[str, ModelResult]:
    """Fit OLS, ridge, LASSO, and logistic models for comparison."""

    X = frame[feature_columns()]
    y = frame["adjusted_rating"]
    y_like = frame["liked"]
    X_train, X_test, y_train, y_test, like_train, like_test = train_test_split(
        X, y, y_like, test_size=0.25, random_state=random_state
    )

    return {
        "OLS": fit_ols(X_train, y_train, X_test, y_test),
        "Ridge": fit_ridge(X_train, y_train, X_test, y_test),
        "LASSO": fit_lasso(X_train, y_train, X_test, y_test),
        "Logistic": fit_logistic(X_train, like_train, X_test, like_test),
    }


def fit_ols(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    X_design = _with_intercept(X_train.to_numpy())
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_train.to_numpy()
    preds = _with_intercept(X_test.to_numpy()) @ beta
    return _regression_result("OLS", beta[0], beta[1:], X_train.columns, y_test, preds)


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 24)),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    ridge = model.named_steps["ridgecv"]
    coefs = ridge.coef_ / model.named_steps["standardscaler"].scale_
    intercept = float(ridge.intercept_ - np.dot(model.named_steps["standardscaler"].mean_, coefs))
    return _regression_result("Ridge", intercept, coefs, X_train.columns, y_test, preds)


def fit_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=5, random_state=7, max_iter=10_000),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    lasso = model.named_steps["lassocv"]
    coefs = lasso.coef_ / model.named_steps["standardscaler"].scale_
    intercept = float(lasso.intercept_ - np.dot(model.named_steps["standardscaler"].mean_, coefs))
    return _regression_result("LASSO", intercept, coefs, X_train.columns, y_test, preds)


def fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000),
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(float)
    logistic = model.named_steps["logisticregression"]
    coefs = logistic.coef_[0] / model.named_steps["standardscaler"].scale_
    intercept = float(
        logistic.intercept_[0] - np.dot(model.named_steps["standardscaler"].mean_, coefs)
    )
    return _regression_result("Logistic", intercept, coefs, X_train.columns, y_test, preds)


def naive_prediction_interval(
    frame: pd.DataFrame,
    candidate_features: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Approximate OLS prediction intervals for candidate residual predictions."""

    X = frame[feature_columns()].to_numpy()
    y = frame["adjusted_rating"].to_numpy()
    X_design = _with_intercept(X)
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    residuals = y - X_design @ beta
    df = max(len(y) - X_design.shape[1], 1)
    sigma = np.sqrt(np.sum(residuals**2) / df)
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    X0 = _with_intercept(candidate_features[feature_columns()].to_numpy())
    leverage = np.sum((X0 @ xtx_inv) * X0, axis=1)
    t_value = stats.t.ppf(1 - alpha / 2, df)
    residual_pred = X0 @ beta
    margin = t_value * sigma * np.sqrt(1 + leverage)
    return pd.DataFrame(
        {
            "residual_pred": residual_pred,
            "naive_low": residual_pred - margin,
            "naive_high": residual_pred + margin,
            "naive_se": margin / t_value,
        },
        index=candidate_features.index,
    )


def cluster_bootstrap_predictions(
    frame: pd.DataFrame,
    candidate_features: pd.DataFrame,
    n_bootstrap: int = 80,
    random_state: int = 7,
) -> pd.DataFrame:
    """Cluster bootstrap by user to respect within-user rating correlation."""

    rng = np.random.default_rng(random_state)
    users = frame["user_id"].unique()
    candidate_X = _with_intercept(candidate_features[feature_columns()].to_numpy())
    predictions = []

    for _ in range(n_bootstrap):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        sample = pd.concat([frame[frame["user_id"] == user] for user in sampled_users])
        X = _with_intercept(sample[feature_columns()].to_numpy())
        y = sample["adjusted_rating"].to_numpy()
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        predictions.append(candidate_X @ beta)

    boot = np.vstack(predictions)
    return pd.DataFrame(
        {
            "bootstrap_low": np.percentile(boot, 2.5, axis=0),
            "bootstrap_high": np.percentile(boot, 97.5, axis=0),
            "bootstrap_se": boot.std(axis=0),
        },
        index=candidate_features.index,
    )


def coefficient_intervals(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Naive OLS coefficient intervals for the explanation dashboard."""

    X = _with_intercept(frame[feature_columns()].to_numpy())
    y = frame["adjusted_rating"].to_numpy()
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    df = max(len(y) - X.shape[1], 1)
    sigma2 = np.sum(residuals**2) / df
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_value = stats.t.ppf(1 - alpha / 2, df)
    names = ["intercept"] + feature_columns()
    return pd.DataFrame(
        {
            "feature": names,
            "estimate": beta,
            "std_error": se,
            "low": beta - t_value * se,
            "high": beta + t_value * se,
            "p_value": 2 * (1 - stats.t.cdf(np.abs(beta / np.maximum(se, 1e-12)), df)),
        }
    )


def _regression_result(
    name: str,
    intercept: float,
    coefs: np.ndarray,
    columns: pd.Index,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> ModelResult:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    residuals = y_true.to_numpy() - y_pred
    residual_std = float(np.std(residuals, ddof=1))
    n = len(y_true)
    k = len(coefs) + 1
    rss = max(float(np.sum(residuals**2)), 1e-12)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    return ModelResult(
        name=name,
        coefficients=pd.Series(coefs, index=columns).sort_values(key=np.abs, ascending=False),
        intercept=float(intercept),
        rmse=rmse,
        mae=mae,
        mse=float(mse),
        aic=float(2 * k - 2 * log_likelihood),
        bic=float(k * np.log(n) - 2 * log_likelihood),
        residual_std=residual_std,
    )


def _with_intercept(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(values)), values])

