import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

DATA_DIR = Path("dataset")
PLOT_DIR = Path("plts")
MODEL_DIR = Path("model")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_FILE = DATA_DIR / "X_train.csv"
Y_TRAIN_FILE = DATA_DIR / "y_train.csv"
X_TEST_FILE = DATA_DIR / "X_test.csv"

# Optional: preprocessing does not currently create y_test.csv.
Y_TEST_FILE = DATA_DIR / "y_test.csv"
METADATA_TEST_FILE = DATA_DIR / "metadata_test.csv"

RANDOM_STATE = 42
CV_SPLITS = 4
N_JOBS = -1

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# ============================================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================================

for file_path in (X_TRAIN_FILE, Y_TRAIN_FILE, X_TEST_FILE):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Não foi encontrado {file_path}. "
            "Executa primeiro o script de pré-processamento atualizado."
        )

X_train = pd.read_csv(X_TRAIN_FILE)
Y_train = pd.read_csv(Y_TRAIN_FILE)
X_test = pd.read_csv(X_TEST_FILE)

TARGET_COLUMNS = [
    "Target_Direction_Sin",
    "Target_Direction_Cos",
    "ShapeElongation",
    "ShapeCircularity",
    "ShapeSolidity",
    "Target_ShapeOrientation_Sin",
    "Target_ShapeOrientation_Cos",
]

missing_targets = [c for c in TARGET_COLUMNS if c not in Y_train.columns]
if missing_targets:
    raise ValueError(
        f"Faltam targets em {Y_TRAIN_FILE}: {missing_targets}. "
        "O ficheiro y_train.csv deve ser produzido pelo pré-processamento atualizado."
    )

Y_train = Y_train[TARGET_COLUMNS].copy()

# Guarantee exactly the same feature columns and order in train and test.
X_test = X_test.reindex(columns=X_train.columns)

# Convert all inputs and targets to numeric; invalid values become NaN.
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")
Y_train = Y_train.apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
Y_train = Y_train.replace([np.inf, -np.inf], np.nan)

# Remove features that contain no usable values in training.
usable_features = X_train.columns[X_train.notna().any()].tolist()
X_train = X_train[usable_features]
X_test = X_test.reindex(columns=usable_features)

if X_train.shape[1] == 0:
    raise ValueError("X_train não contém features numéricas utilizáveis.")

# Rows with missing target values cannot be used for supervised training.
valid_target_rows = Y_train.notna().all(axis=1)
removed_rows = int((~valid_target_rows).sum())
if removed_rows:
    print(f"Rows removed due to missing target values: {removed_rows}")

X_train = X_train.loc[valid_target_rows].reset_index(drop=True)
Y_train = Y_train.loc[valid_target_rows].reset_index(drop=True)

if len(X_train) < CV_SPLITS:
    raise ValueError(
        f"Existem apenas {len(X_train)} linhas válidas; "
        f"são necessárias pelo menos {CV_SPLITS} para CV={CV_SPLITS}."
    )

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Number of input features:", len(usable_features))
print("Targets:", TARGET_COLUMNS)

# Report whether the new source-derived feature groups reached the model.
source_feature_prefixes = (
    "Poly2D", "Poly3D", "SimCentroid_", "SimulationReferenceImage_hash_"
)
source_features = [
    c for c in X_train.columns
    if c.startswith(source_feature_prefixes)
    or c == "SimulationReferenceImage_missing"
]
print(f"Features derived from the four requested source fields: {len(source_features)}")
print(source_features)


# ============================================================================
# 3. MODEL AND HYPERPARAMETER SEARCH
# ============================================================================

# Imputation is fitted inside each CV fold, avoiding validation-fold leakage.
# XGBoost can handle NaNs itself, but the imputer also makes the pipeline
# robust to different XGBoost versions and missing-value patterns.
base_estimator = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=RANDOM_STATE,
    n_jobs=1,
    tree_method="hist",
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", MultiOutputRegressor(base_estimator, n_jobs=N_JOBS)),
])

# Deliberately smaller than the original exhaustive grid:
# the previous 4^7 grid would require 16,384 parameter combinations,
# multiplied by CV folds and seven target estimators. This grid has 32
# combinations (32 x CV folds x seven target estimators).
param_grid = {
    "model__estimator__max_depth": [3, 6],
    "model__estimator__learning_rate": [0.03, 0.07],
    "model__estimator__n_estimators": [200, 400],
    "model__estimator__subsample": [0.8, 1.0],
    "model__estimator__colsample_bytree": [0.8, 1.0],
}

cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    n_jobs=N_JOBS,
    cv=cv,
    verbose=1,
    refit=True,
)

print("\nStarting GridSearchCV...")
grid_search.fit(X_train, Y_train)

print("\nBest parameters:")
print(grid_search.best_params_)
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))

best_model = grid_search.best_estimator_


# ============================================================================
# 4. PREDICTIONS
# ============================================================================

y_pred = best_model.predict(X_test)
predictions = pd.DataFrame(y_pred, columns=TARGET_COLUMNS)

# Direction: 360-degree periodic variable.
pred_sin_dir = predictions["Target_Direction_Sin"].to_numpy()
pred_cos_dir = predictions["Target_Direction_Cos"].to_numpy()
pred_direction_deg = (
    np.degrees(np.arctan2(pred_sin_dir, pred_cos_dir)) + 360
) % 360

# Orientation: axial variable with a 180-degree period.
pred_sin_ori = predictions["Target_ShapeOrientation_Sin"].to_numpy()
pred_cos_ori = predictions["Target_ShapeOrientation_Cos"].to_numpy()
pred_orientation_deg = (
    np.degrees(0.5 * np.arctan2(pred_sin_ori, pred_cos_ori)) + 180
) % 180

results = pd.DataFrame({
    "pred_direction_deg": pred_direction_deg,
    "pred_orientation_deg": pred_orientation_deg,
    "pred_shape_elongation": predictions["ShapeElongation"],
    "pred_shape_circularity": predictions["ShapeCircularity"],
    "pred_shape_solidity": predictions["ShapeSolidity"],
})

# Save raw encoded predictions as well as reconstructed values.
for col in TARGET_COLUMNS:
    results[col] = predictions[col]

results.to_csv(DATA_DIR / "predictions_test.csv", index=False)
print("\nPredictions saved to:", DATA_DIR / "predictions_test.csv")


# ============================================================================
# 5. OPTIONAL TEST EVALUATION
# ============================================================================

def circular_absolute_error(true_values, pred_values, period):
    """Shortest absolute angular distance for a given angular period."""
    delta = np.abs(np.asarray(true_values) - np.asarray(pred_values)) % period
    return np.minimum(delta, period - delta)


def print_regression_metrics(y_true, y_hat, label):
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_hat)
    if valid.sum() == 0:
        print(f"{label}: no valid paired values.")
        return
    print(
        f"{label}: MAE={mean_absolute_error(y_true[valid], y_hat[valid]):.6f}, "
        f"RMSE={np.sqrt(mean_squared_error(y_true[valid], y_hat[valid])):.6f}, "
        f"R2={r2_score(y_true[valid], y_hat[valid]):.6f}"
    )


if Y_TEST_FILE.exists():
    Y_test = pd.read_csv(Y_TEST_FILE)
    missing_test_targets = [c for c in TARGET_COLUMNS if c not in Y_test.columns]
    if missing_test_targets:
        raise ValueError(
            f"{Y_TEST_FILE} existe, mas faltam targets: {missing_test_targets}"
        )

    Y_test = Y_test[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if len(Y_test) != len(predictions):
        raise ValueError(
            "O número de linhas de y_test.csv não coincide com X_test.csv. "
            "Confirma o alinhamento no pré-processamento."
        )

    print("\n--- Test metrics: encoded regression targets ---")
    for col in TARGET_COLUMNS:
        print_regression_metrics(Y_test[col], predictions[col], col)

    # Reconstruct true angles from encoded targets.
    true_direction_deg = (
        np.degrees(np.arctan2(
            Y_test["Target_Direction_Sin"],
            Y_test["Target_Direction_Cos"]
        )) + 360
    ) % 360

    true_orientation_deg = (
        np.degrees(0.5 * np.arctan2(
            Y_test["Target_ShapeOrientation_Sin"],
            Y_test["Target_ShapeOrientation_Cos"]
        )) + 180
    ) % 180

    direction_error = circular_absolute_error(
        true_direction_deg, pred_direction_deg, 360
    )
    orientation_error = circular_absolute_error(
        true_orientation_deg, pred_orientation_deg, 180
    )

    print("\n--- Test angular errors ---")
    print("Direction error (degrees):")
    print(pd.Series(direction_error).describe())
    print("\nOrientation error (degrees):")
    print(pd.Series(orientation_error).describe())

    evaluation = pd.DataFrame({
        "true_direction_deg": true_direction_deg,
        "pred_direction_deg": pred_direction_deg,
        "direction_error_deg": direction_error,
        "true_orientation_deg": true_orientation_deg,
        "pred_orientation_deg": pred_orientation_deg,
        "orientation_error_deg": orientation_error,
    })
    evaluation.to_csv(DATA_DIR / "evaluation_test.csv", index=False)

else:
    print(
        "\nNo y_test.csv found. Test predictions were generated, but supervised "
        "test metrics cannot be calculated without ground-truth targets."
    )

    # The current preprocessing exports these true angles in metadata_test.csv.
    # This allows angular evaluation when those columns are present.
    if METADATA_TEST_FILE.exists():
        metadata_test = pd.read_csv(METADATA_TEST_FILE)
        if len(metadata_test) != len(predictions):
            print(
                "WARNING: metadata_test.csv row count differs from X_test.csv; "
                "skipping metadata-based evaluation to avoid misalignment."
            )
        else:
            eval_data = {}
            if "FinalGrowthAngleDeg" in metadata_test.columns:
                true_direction = pd.to_numeric(
                    metadata_test["FinalGrowthAngleDeg"], errors="coerce"
                )
                valid = true_direction.notna().to_numpy()
                if valid.any():
                    eval_data["true_direction_deg"] = true_direction
                    eval_data["pred_direction_deg"] = pred_direction_deg
                    direction_error = circular_absolute_error(
                        true_direction, pred_direction_deg, 360
                    )
                    print("\nDirection angular error (degrees):")
                    print(pd.Series(direction_error[valid]).describe())

            if "ShapeOrientationDeg" in metadata_test.columns:
                true_orientation = pd.to_numeric(
                    metadata_test["ShapeOrientationDeg"], errors="coerce"
                )
                valid = true_orientation.notna().to_numpy()
                if valid.any():
                    orientation_error = circular_absolute_error(
                        true_orientation, pred_orientation_deg, 180
                    )
                    print("\nOrientation angular error (degrees):")
                    print(pd.Series(orientation_error[valid]).describe())

            if eval_data:
                pd.DataFrame(eval_data).to_csv(
                    DATA_DIR / "angular_evaluation_test.csv", index=False
                )


# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================

# MultiOutputRegressor contains one fitted XGBRegressor per target.
# Plot importances averaged across the seven fitted estimators.
fitted_multioutput = best_model.named_steps["model"]
importance_arrays = [
    estimator.feature_importances_
    for estimator in fitted_multioutput.estimators_
]
mean_importance = np.mean(np.vstack(importance_arrays), axis=0)

feature_importance = pd.DataFrame({
    "feature": usable_features,
    "importance_mean": mean_importance,
}).sort_values("importance_mean", ascending=False)

feature_importance.to_csv(DATA_DIR / "feature_importance.csv", index=False)

top_n = min(25, len(feature_importance))
plt.figure(figsize=(10, max(6, top_n * 0.28)))
top_features = feature_importance.head(top_n).iloc[::-1]
plt.barh(top_features["feature"], top_features["importance_mean"])
plt.xlabel("Mean feature importance across targets")
plt.title("XGBoost feature importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance_multioutput.png", dpi=200)
plt.close()


# ============================================================================
# 7. SAVE MODEL PACKAGE
# ============================================================================

package = {
    "model": best_model,
    "metadata": {
        "target_columns": TARGET_COLUMNS,
        "direction_encoding": "sin(theta), cos(theta), period 360 degrees",
        "direction_reconstruction": "(degrees(atan2(sin, cos)) + 360) % 360",
        "orientation_encoding": "sin(2*theta), cos(2*theta), period 180 degrees",
        "orientation_reconstruction": "(degrees(0.5*atan2(sin2, cos2)) + 180) % 180",
        "shape_targets": [
            "ShapeElongation",
            "ShapeCircularity",
            "ShapeSolidity",
        ],
        "feature_columns": usable_features,
        "source_derived_features": source_features,
        "best_params": grid_search.best_params_,
        "best_cv_neg_mean_squared_error": float(grid_search.best_score_),
        "random_state": RANDOM_STATE,
        "cv": f"KFold(n_splits={CV_SPLITS}, shuffle=True)",
    },
}

model_path = MODEL_DIR / "corrosion_multioutput_model.pkl"
joblib.dump(package, model_path)

print("\n" + "=" * 65)
print("TRAINING COMPLETED")
print("=" * 65)
print("Model saved to:", model_path)
print("Feature importance saved to:", DATA_DIR / "feature_importance.csv")
print("Plots saved to:", PLOT_DIR)
