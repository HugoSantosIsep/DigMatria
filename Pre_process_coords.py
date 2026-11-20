
import re
from pathlib import Path

import numpy as np
import pandas as pd

from shapely.geometry import Polygon, MultiPolygon, MultiPoint


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

TRAIN_CSV = "consolidado_metricas_2016_2025.csv"

TEST_CSV = (
    "obra/2026/01_01_2026/pred/ps/"
    "metricas_anomalias_unicas_2026-01-01_ambiente_3D_HIBRIDO.csv"
)

OUT_DIR = Path("dataset")

OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================================
# 2. LOAD DATA
# ============================================================================

df_training = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

print("Training original:", df_training.shape)
print("Test original:", df_test.shape)


# ============================================================================
# 3. FILTER CORROSION
# ============================================================================

df_training = df_training[
    df_training["classe_nome"] == "corrosion"
].copy()

df_test = df_test[
    df_test["classe_nome"] == "corrosion"
].copy()

print("Training corrosion:", df_training.shape)
print("Test corrosion:", df_test.shape)


# ============================================================================
# 4. DROP UNNECESSARY COLUMNS
# ============================================================================

list_drop_features = [
    "area_m2", "area_cm2", "qa_note",
    "percent_do_total_%", "n_hit_points",
    "n_area_points_used", "image_paths",
    "n_images", "Coordinates",
    "n_detections_merged", "det_ids_merged",
    "area_method", "classe", "local_spacing_m",
    "conf_mean", "conf_max",
    "Projection2DStatus", "Polygon3DStatus",
    "SurfaceGrowthMethod", "Initial3DSeedSource",
    "images", "SurfaceSpacing_m",
    "SurfacePatchVertices", "MaxTransitionUsedDeg",
    "MeanTransitionUsedDeg", "AddedVertices3D",
    "MeanFrontierExposure", "GeometrySupportCoverage",
    "SurfaceVertexCount", "SurfaceNormalX",
    "SurfaceNormalY", "SurfaceNormalZ",
    "SurfaceAzimuthDeg", "MorphIterations",
    "SimulationDate", "Surface3DArea_cm2",
    "Added3DArea_cm2", "ObservedArea_cm2",
    "SimulationReferenceDetID", "classe_nome"
]

# Keep dano_id temporarily for identification/grouping.
# Remove it only from the model features.

df_training = df_training.drop(columns=list_drop_features, errors="ignore")
df_test = df_test.drop(columns=list_drop_features, errors="ignore")


# Parse dates before any temporal operation.
for _df in (df_training, df_test):
    if "data" not in _df.columns:
        raise ValueError("A coluna 'data' é necessária para criar lag features.")
    if "SimulationReferenceImage" not in _df.columns:
        raise ValueError("A coluna 'SimulationReferenceImage' é necessária para o agrupamento temporal.")
    _df["data"] = pd.to_datetime(_df["data"], errors="coerce")

df_training = df_training.sort_values(
    ["data", "SimulationReferenceImage"], kind="mergesort", na_position="last"
).reset_index(drop=True)
df_test = df_test.sort_values(
    ["data", "SimulationReferenceImage"], kind="mergesort", na_position="last"
).reset_index(drop=True)


# ============================================================================
# 5. GEOMETRY FEATURE EXTRACTION
# ============================================================================

def extract_shape_features(polygon_string):

    empty_result = {
        "ShapeArea": np.nan,
        "ShapePerimeter": np.nan,
        "ShapeElongation": np.nan,
        "ShapeCircularity": np.nan,
        "ShapeSolidity": np.nan,
        "ShapeOrientationDeg": np.nan,
        "ShapeWidth": np.nan,
        "ShapeHeight": np.nan
    }

    if not isinstance(polygon_string, str):
        return empty_result

    # Extract 2D coordinates from the polygon string
    points = re.findall(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*\]",
        polygon_string
    )

    if len(points) < 3:
        return empty_result

    coords = np.array(points, dtype=float)

    # Remove only consecutive duplicate points, preserving contour order.
    if len(coords) > 1:
        keep = np.r_[True, np.any(np.diff(coords, axis=0) != 0, axis=1)]
        coords = coords[keep]
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    if len(coords) < 3:
        return empty_result

    try:

        polygon = Polygon(coords)

        # Repair invalid geometry
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty or polygon.area <= 0:
            return empty_result

        # If geometry has multiple components, retain all components
        if isinstance(polygon, MultiPolygon):
            geometry = polygon
        else:
            geometry = polygon

        # Convex hull
        hull = geometry.convex_hull

        # Geometry measurements
        area = geometry.area
        perimeter = geometry.length

        circularity = (
            4 * np.pi * area / perimeter**2
            if perimeter > 0 else np.nan
        )

        solidity = (
            area / hull.area
            if hull.area > 0 else np.nan
        )

        # Collect exterior coordinates
        if isinstance(geometry, MultiPolygon):
            all_coords = np.vstack([
                np.asarray(part.exterior.coords)
                for part in geometry.geoms
            ])
        else:
            all_coords = np.asarray(
                geometry.exterior.coords
            )

        # PCA for principal orientation
        centered = all_coords - all_coords.mean(axis=0)

        covariance = np.cov(centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        eigenvalues = np.maximum(eigenvalues, 1e-10)

        idx = np.argmax(eigenvalues)

        direction = eigenvectors[:, idx]

        orientation = np.degrees(
            np.arctan2(direction[1], direction[0])
        ) % 180

        elongation = np.sqrt(
            eigenvalues.max() / eigenvalues.min()
        )

        # Bounding box
        min_x, min_y, max_x, max_y = geometry.bounds

        return {
            "ShapeArea": area,
            "ShapePerimeter": perimeter,
            "ShapeElongation": elongation,
            "ShapeCircularity": circularity,
            "ShapeSolidity": solidity,
            "ShapeOrientationDeg": orientation,
            "ShapeWidth": max_x - min_x,
            "ShapeHeight": max_y - min_y
        }

    except Exception:
        return empty_result


# Extract geometry from training
if "SimulatedPolygon2D" in df_training.columns:

    shape_train = pd.DataFrame(
        df_training["SimulatedPolygon2D"].apply(
            extract_shape_features
        ).tolist(),
        index=df_training.index
    )

    df_training = pd.concat(
        [df_training, shape_train],
        axis=1
    )


# Extract geometry from test, only if available
if "SimulatedPolygon2D" in df_test.columns:

    shape_test = pd.DataFrame(
        df_test["SimulatedPolygon2D"].apply(
            extract_shape_features
        ).tolist(),
        index=df_test.index
    )

    df_test = pd.concat(
        [df_test, shape_test],
        axis=1
    )


# ============================================================================
# 7. ANGULAR FEATURE ENCODING
# ============================================================================

def encode_direction(df, column):

    if column not in df.columns:
        return df

    angle = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    radians = np.deg2rad(angle)

    df[column + "_sin"] = np.sin(radians)
    df[column + "_cos"] = np.cos(radians)

    return df


def encode_orientation(df, column):

    if column not in df.columns:
        return df

    angle = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # Axial orientation: period of 180 degrees
    radians = np.deg2rad(2 * angle)

    df[column + "_sin2"] = np.sin(radians)
    df[column + "_cos2"] = np.cos(radians)

    return df


# Environmental directions (360 degrees)
for column in [
    "WindDirectionDeg",
    "WDRDirectionDeg"
]:

    df_training = encode_direction(df_training, column)
    df_test = encode_direction(df_test, column)


# Initial shape orientation (180 degrees)
df_training = encode_orientation(
    df_training, "OriginalShapeAngleDeg"
)

df_test = encode_orientation(
    df_test, "OriginalShapeAngleDeg"
)


# ============================================================================
# 8. CREATE LAG FEATURES (TRAIN + TEST IN ONE CHRONOLOGICAL SEQUENCE)
# ============================================================================

GROUP_COL = "SimulationReferenceImage"  # metadata, used internally for grouping

LAG_SOURCE_COLUMNS = [
    "EnvironmentFactor",
    "WindDirectionDeg_sin", "WindDirectionDeg_cos",
    "WDRDirectionDeg_sin", "WDRDirectionDeg_cos",
    "ProximityFactor", "DistanceWater_m", "DistanceDamage_m",
    "CorrosionNeighbors", "OriginalElongationRatio",
    "OriginalShapeAngleDeg_sin2", "OriginalShapeAngleDeg_cos2",
    "ShapeArea", "ShapePerimeter", "ShapeElongation",
    "ShapeCircularity", "ShapeSolidity", "ShapeOrientationDeg",
    "ShapeWidth", "ShapeHeight",
    "centroid_X", "centroid_Y", "centroid_Z",
]
LAG_SOURCE_COLUMNS = list(dict.fromkeys(LAG_SOURCE_COLUMNS))
LAG_STEPS = [1]

_train = df_training.copy()
_test = df_test.copy()
_train["__dataset_split"] = "train"
_test["__dataset_split"] = "test"
_train["__split_order"] = 0
_test["__split_order"] = 1

_all = pd.concat([_train, _test], ignore_index=True, sort=False)
_all["data"] = pd.to_datetime(_all["data"], errors="coerce")
_all = _all.sort_values(
    [GROUP_COL, "data", "__split_order"],
    kind="mergesort", na_position="last"
).reset_index(drop=True)

_available_lag_columns = [c for c in LAG_SOURCE_COLUMNS if c in _all.columns]
for col in _available_lag_columns:
    _all[col] = pd.to_numeric(_all[col], errors="coerce")

lag_columns = []
valid_group = _all[GROUP_COL].notna()
for col in _available_lag_columns:
    for lag_step in LAG_STEPS:
        lag_col = f"{col}_lag_{lag_step}"
        if lag_col in _all.columns:
            raise ValueError(f"A coluna de lag já existe: {lag_col}")
        _all[lag_col] = np.nan
        _all.loc[valid_group, lag_col] = (
            _all.loc[valid_group]
            .groupby(GROUP_COL, sort=False)[col]
            .shift(lag_step)
        )
        lag_columns.append(lag_col)

if lag_columns:
    _all[lag_columns] = _all[lag_columns].fillna(0)

df_training = (
    _all.loc[_all["__dataset_split"] == "train"]
    .drop(columns=["__dataset_split", "__split_order"])
    .sort_values(["data", GROUP_COL], kind="mergesort", na_position="last")
    .reset_index(drop=True)
)
df_test = (
    _all.loc[_all["__dataset_split"] == "test"]
    .drop(columns=["__dataset_split", "__split_order"])
    .sort_values(["data", GROUP_COL], kind="mergesort", na_position="last")
    .reset_index(drop=True)
)

print("Lag features created:", len(lag_columns))
print("Lag source columns:", _available_lag_columns)
print("Training dates:", df_training["data"].min(), "->", df_training["data"].max())
print("Test dates:", df_test["data"].min(), "->", df_test["data"].max())


# ============================================================================
# 9. CREATE TARGET VARIABLES
# ============================================================================

# Target 1: Final growth direction (360 degrees)

if "FinalGrowthAngleDeg" not in df_training.columns:
    raise ValueError(
        "FinalGrowthAngleDeg is missing from training data."
    )

final_angle = pd.to_numeric(
    df_training["FinalGrowthAngleDeg"],
    errors="coerce"
)

final_angle_rad = np.deg2rad(final_angle)

df_training["Target_Direction_Sin"] = np.sin(
    final_angle_rad
)

df_training["Target_Direction_Cos"] = np.cos(
    final_angle_rad
)


# Target 2: Shape characteristics

shape_target_columns = [
    "ShapeElongation",
    "ShapeCircularity",
    "ShapeSolidity"
]

# Target 3: Principal shape orientation (180 degrees)

if "ShapeOrientationDeg" in df_training.columns:

    orientation_rad = np.deg2rad(
        2 * df_training["ShapeOrientationDeg"]
    )

    df_training["Target_ShapeOrientation_Sin"] = (
        np.sin(orientation_rad)
    )

    df_training["Target_ShapeOrientation_Cos"] = (
        np.cos(orientation_rad)
    )


# ============================================================================
# 10. DEFINE TARGETS
# ============================================================================

target_columns = [
    "Target_Direction_Sin",
    "Target_Direction_Cos",
    "ShapeElongation",
    "ShapeCircularity",
    "ShapeSolidity",
    "Target_ShapeOrientation_Sin",
    "Target_ShapeOrientation_Cos"
]

# Verify targets
missing_targets = [
    col for col in target_columns
    if col not in df_training.columns
]

if missing_targets:
    raise ValueError(
        f"Missing target columns: {missing_targets}"
    )


# Remove observations without valid targets
df_training = df_training.dropna(
    subset=target_columns
).copy()


# ============================================================================
# 11. DEFINE METADATA AND EXCLUDED FEATURES
# ============================================================================

# Requested fields are preserved in metadata and excluded from model inputs.
metadata_columns = [
    "dano_id",
    "data",
    "3D coordinates of polygons",
    "SimulatedPolygon2D",
    "SimulationCentroidXYZ",
    "SimulationReferenceImage",
    "FinalGrowthAngleDeg",
    "OriginalShapeAngleDeg",
    "ShapeOrientationDeg",
]
metadata_train_columns = [c for c in metadata_columns if c in df_training.columns]
metadata_test_columns = [c for c in metadata_columns if c in df_test.columns]
metadata_train = df_training[metadata_train_columns].copy()
metadata_test = df_test[metadata_test_columns].copy()

# Current geometry defines shape targets, so only historical lag versions
# are eligible as model inputs.
current_geometry_columns = [
    "ShapeArea", "ShapePerimeter", "ShapeElongation",
    "ShapeCircularity", "ShapeSolidity", "ShapeOrientationDeg",
    "ShapeWidth", "ShapeHeight",
]
excluded_columns = list(dict.fromkeys(
    target_columns + metadata_columns + current_geometry_columns
    + ["WindDirectionDeg", "WDRDirectionDeg"]
))


# ============================================================================
# 12. CREATE X_train, y_train AND X_test
# ============================================================================

y_train = df_training[target_columns].copy()
X_train = df_training.drop(columns=excluded_columns, errors="ignore")
X_test = df_test.drop(columns=excluded_columns, errors="ignore")

X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

valid_columns = X_train.columns[X_train.notna().any()].tolist()
X_train = X_train[valid_columns]
X_test = X_test.reindex(columns=valid_columns)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


# ============================================================================
# 13. SAVE DATASETS
# ============================================================================

X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
X_test.to_csv(OUT_DIR / "X_test.csv", index=False)

# ============================================================================
# 14. SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)

print("\nmetadata_train:", metadata_train.shape)
print("metadata_test:", metadata_test.shape)

print("\nFeatures:")
print(X_train.columns.tolist())

print("\nTargets:")
print(y_train.columns.tolist())

print("\nMissing values in X_train:")
print(X_train.isna().sum())

print("\nMissing values in y_train:")
print(y_train.isna().sum())

print("\nFiles saved in:", OUT_DIR.resolve())

