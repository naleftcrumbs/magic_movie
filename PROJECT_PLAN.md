# Magic Movie Project Plan

## Goal

Build a recommendation web app for indecisive movie watchers who want statistically informed recommendations rather than opaque scores.

## Core Question

How can a recommender be statistically interpretable and uncertainty-aware rather than purely predictive?

## Modeling Approach

1. Estimate user, movie, and global baselines.
2. Model residual deviations from baseline preferences.
3. Fit OLS, ridge, LASSO, and logistic models.
4. Compare RMSE, MAE, MSE, AIC, and BIC.
5. Estimate naive OLS intervals.
6. Refit with user-cluster bootstrap intervals to acknowledge non-IID ratings.
7. Turn predictions and uncertainty into decision modes.

## Web Pages

1. Preference input
2. Recommendations
3. Statistical explanation
4. Model comparison dashboard
5. Diagnostics and statistical assumptions

## Next Extensions

- Add a real MovieLens CSV loader.
- Persist user preference sessions.
- Add Q-Q plots and leverage plots as rendered image endpoints.
- Add ridge coefficient paths and RMSE-vs-lambda curves.
- Deploy with a small API framework once the course demo is stable.

