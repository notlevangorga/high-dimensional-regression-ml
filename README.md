# High-Dimensional Regression for Health Analytics

## Overview
This project explores applied regression modeling in a high-dimensional health dataset, with an emphasis on **model robustness, regularization, and generalization** rather than feature interpretation or causal claims.

The goal is to evaluate how different regression approaches perform when the number of numerical features is large and multicollinearity is likely.

---

## Approach
A unified machine learning pipeline was built to compare multiple regression models under the same preprocessing and evaluation framework:

- **Ridge Regression** (L2 regularization)
- **Lasso Regression** (L1 regularization)
- **Gradient Boosting Regressor** (non-linear ensemble model)

Key characteristics of the approach:
- Numeric feature selection to avoid high-cardinality categorical expansion
- Mean imputation for missing values
- Feature scaling for linear models
- Cross-validation for regularized models
- Consistent train/test evaluation using Mean Squared Error (MSE)

---

## Model Selection
Models are compared using cross-validated MSE where applicable, with final selection based on generalization performance.  
The best-performing model is persisted for reuse.

---

## Outputs
- **Trained model:** `models/best_model.joblib`
- **Model comparison results:** `outputs/model_comparison_results.csv`

---

## Project Structure
