"""
High-Dimensional Regression Pipeline
------------------------------------

This project evaluates multiple regression models on a high-dimensional
health dataset, focusing on robustness, scalability, and interpretability.

Models evaluated:
- Ridge Regression
- Lasso Regression
- Gradient Boosting Regressor

Evaluation metric:
- Mean Squared Error (MSE)

Outputs:
- Trained best model (saved to /models)
- Model comparison results (saved to /outputs)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor


# =======================
# Configuration
# =======================

DATA_PATH = "data/training_dataset.xlsx"
TARGET_COL = "WEIGHTLBTC_A"

MODEL_OUTPUT_PATH = "models/best_model.joblib"
RESULTS_OUTPUT_PATH = "outputs/model_comparison_results.csv"

RANDOM_STATE = 42


# =======================
# 1. Load & prepare data
# =======================

print("Loading dataset...")
df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=[TARGET_COL])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("Selecting numeric features only...")
X = X.select_dtypes(include=[np.number])


# =======================
# 2. Train / test split
# =======================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# =======================
# 3. Preprocessing pipeline
# =======================

preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


# =======================
# 4. Model definitions
# =======================

models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=10000),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
}

param_grids = {
    "Ridge": {"model__alpha": [0.1, 1, 10]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1]},
    "GradientBoosting": None
}


# =======================
# 5. Training & evaluation
# =======================

results = []
best_model = None
best_score = np.inf
best_predictions = None
best_model_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    if param_grids[name] is None:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        test_mse = mean_squared_error(y_test, preds)

        results.append({
            "Model": name,
            "CV_MSE": np.nan,
            "Test_MSE": test_mse
        })

        score_for_selection = test_mse

    else:
        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            scoring="neg_mean_squared_error",
            cv=2,
            n_jobs=1
        )

        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)

        cv_mse = -grid.best_score_
        test_mse = mean_squared_error(y_test, preds)

        results.append({
            "Model": name,
            "CV_MSE": cv_mse,
            "Test_MSE": test_mse
        })

        score_for_selection = cv_mse
        pipeline = grid.best_estimator_

    if score_for_selection < best_score:
        best_score = score_for_selection
        best_model = pipeline
        best_predictions = preds
        best_model_name = name

    print(f"{name} completed.")


# =======================
# 6. Results summary
# =======================

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df.to_string(index=False))

print(f"\nSelected best model: {best_model_name}")


# =======================
# 7. Visual diagnostics
# =======================

plt.figure()
plt.bar(results_df["Model"], results_df["Test_MSE"])
plt.ylabel("Test MSE")
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(y_test, best_predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.tight_layout()
plt.show()

residuals = y_test - best_predictions
plt.figure()
plt.scatter(best_predictions, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title(f"Residual Analysis ({best_model_name})")
plt.tight_layout()
plt.show()


# =======================
# 8. Save outputs
# =======================

joblib.dump(best_model, MODEL_OUTPUT_PATH)
results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)

print("\nSaved artifacts:")
print(f"- {MODEL_OUTPUT_PATH}")
print(f"- {RESULTS_OUTPUT_PATH}")
