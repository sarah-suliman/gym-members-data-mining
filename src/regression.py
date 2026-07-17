"""
Regression analysis for the Gym Members Exercise Dataset.

This script predicts calories burned using demographic, physiological,
and workout-related variables. It compares a mean-value baseline,
Linear Regression, and Random Forest Regressor.

Compatible with Python 3.9.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gym_members_exercise_cleaned.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

METRICS_PATH = RESULTS_DIR / "regression_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "regression_predictions.csv"
FEATURE_IMPORTANCE_PATH = (
    RESULTS_DIR / "regression_feature_importance.csv"
)

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_regressor.joblib"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression_model.joblib"
RANDOM_FOREST_MODEL_PATH = (
    MODELS_DIR / "random_forest_regressor.joblib"
)


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Calories_Burned"

NUMERIC_FEATURES = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Heart_Rate_Reserve",
    "Session_Duration (hours)",
    "Fat_Percentage",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    "Experience_Level",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Workout_Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

sns.set_theme(
    context="notebook",
    style="whitegrid",
)


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned gym member dataset.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found at: {}\n"
            "Run src/preprocessing.py before regression.".format(path)
        )

    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Confirm that all regression features and the target are available.
    """
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: {}".format(
                missing_columns
            )
        )

    missing_value_count = int(
        df[required_columns].isna().sum().sum()
    )

    if missing_value_count > 0:
        raise ValueError(
            "Regression data contains missing values. "
            "Missing-value count: {}".format(missing_value_count)
        )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset into training and testing sets.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing steps for numerical and categorical features.

    Numerical variables are standardized, while categorical variables
    are converted into one-hot encoded indicator columns.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_models() -> Dict[str, Pipeline]:
    """
    Build the baseline and machine learning regression pipelines.
    """
    baseline_model = Pipeline(
        steps=[
            (
                "regressor",
                DummyRegressor(
                    strategy="mean",
                ),
            ),
        ]
    )

    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Baseline": baseline_model,
        "Linear Regression": linear_model,
        "Random Forest": random_forest_model,
    }


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def calculate_rmse(
    actual_values: pd.Series,
    predictions: np.ndarray,
) -> float:
    """
    Calculate root mean squared error.

    This implementation works across scikit-learn versions.
    """
    mse = mean_squared_error(
        actual_values,
        predictions,
    )

    return float(np.sqrt(mse))


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    Dict[str, Union[float, str]],
    pd.Series,
]:
    """
    Fit and evaluate one regression model.
    """
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "MAE": mean_absolute_error(
            y_test,
            predictions,
        ),
        "RMSE": calculate_rmse(
            y_test,
            predictions,
        ),
        "R2": r2_score(
            y_test,
            predictions,
        ),
    }

    prediction_series = pd.Series(
        predictions,
        index=y_test.index,
        name=model_name,
    )

    return metrics, prediction_series


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def create_feature_importance_table(
    model: Pipeline,
) -> pd.DataFrame:
    """
    Extract Random Forest feature importance values.
    """
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": regressor.feature_importances_,
        }
    )

    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False,
        )
        .str.replace(
            "categorical__",
            "",
            regex=False,
        )
    )

    return importance_df.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_current_figure(filename: str) -> Path:
    """
    Save and close the current Matplotlib figure.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved figure: {}".format(output_path.name))

    return output_path


def plot_model_comparison(
    metrics_df: pd.DataFrame,
) -> None:
    """
    Compare regression models using RMSE.
    """
    ordered_metrics = metrics_df.sort_values(
        "RMSE",
        ascending=True,
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=ordered_metrics,
        x="Model",
        y="RMSE",
    )

    plt.title("Regression Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Root Mean Squared Error")

    save_current_figure(
        "regression_model_comparison.png"
    )


def plot_actual_vs_predicted(
    actual_values: pd.Series,
    predictions: pd.Series,
    model_name: str,
) -> None:
    """
    Plot actual calories burned against model predictions.
    """
    minimum_value = min(
        actual_values.min(),
        predictions.min(),
    )

    maximum_value = max(
        actual_values.max(),
        predictions.max(),
    )

    plt.figure(figsize=(8, 7))

    sns.scatterplot(
        x=actual_values,
        y=predictions,
        alpha=0.65,
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    plt.title(
        "{}: Actual vs. Predicted Calories".format(
            model_name
        )
    )
    plt.xlabel("Actual Calories Burned")
    plt.ylabel("Predicted Calories Burned")
    plt.legend()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    save_current_figure(
        "{}_actual_vs_predicted.png".format(safe_name)
    )


def plot_residuals(
    predictions: pd.Series,
    actual_values: pd.Series,
    model_name: str,
) -> None:
    """
    Plot prediction residuals.

    Residual equals actual value minus predicted value.
    """
    residuals = actual_values - predictions

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=predictions,
        y=residuals,
        alpha=0.65,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    plt.title("{} Residual Plot".format(model_name))
    plt.xlabel("Predicted Calories Burned")
    plt.ylabel("Residual")

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    save_current_figure(
        "{}_residual_plot.png".format(safe_name)
    )


def plot_feature_importance(
    importance_df: pd.DataFrame,
) -> None:
    """
    Plot the most influential Random Forest regression features.
    """
    top_features = (
        importance_df
        .head(12)
        .sort_values("Importance")
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
    )

    plt.title(
        "Random Forest Regression Feature Importance"
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    save_current_figure(
        "regression_feature_importance.png"
    )


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_results(
    models: Dict[str, Pipeline],
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    """
    Save regression models and tabular results.
    """
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    joblib.dump(
        models["Baseline"],
        BASELINE_MODEL_PATH,
    )

    joblib.dump(
        models["Linear Regression"],
        LINEAR_MODEL_PATH,
    )

    joblib.dump(
        models["Random Forest"],
        RANDOM_FOREST_MODEL_PATH,
    )

    print("Saved metrics: {}".format(METRICS_PATH.name))
    print(
        "Saved predictions: {}".format(
            PREDICTIONS_PATH.name
        )
    )
    print(
        "Saved feature importance: {}".format(
            FEATURE_IMPORTANCE_PATH.name
        )
    )
    print("Saved model: {}".format(BASELINE_MODEL_PATH.name))
    print("Saved model: {}".format(LINEAR_MODEL_PATH.name))
    print(
        "Saved model: {}".format(
            RANDOM_FOREST_MODEL_PATH.name
        )
    )


# ---------------------------------------------------------------------------
# Main regression pipeline
# ---------------------------------------------------------------------------

def run_regression_analysis() -> None:
    """
    Run the complete calories-burned regression workflow.
    """
    df = load_dataset()
    validate_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(df)

    models = build_models()

    metric_rows = []

    prediction_columns = {
        "Actual_Calories_Burned": y_test,
    }

    fitted_models = {}

    for model_name, model in models.items():
        metrics, predictions = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        metric_rows.append(metrics)

        prediction_columns[
            "{}_Prediction".format(model_name)
        ] = predictions

        fitted_models[model_name] = model

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    predictions_df = pd.DataFrame(
        prediction_columns
    ).sort_index()

    best_model_name = str(
        metrics_df.iloc[0]["Model"]
    )

    best_predictions = prediction_columns[
        "{}_Prediction".format(best_model_name)
    ]

    importance_df = create_feature_importance_table(
        fitted_models["Random Forest"]
    )

    plot_model_comparison(metrics_df)

    plot_actual_vs_predicted(
        actual_values=y_test,
        predictions=best_predictions,
        model_name=best_model_name,
    )

    plot_residuals(
        predictions=best_predictions,
        actual_values=y_test,
        model_name=best_model_name,
    )

    plot_feature_importance(importance_df)

    save_results(
        models=fitted_models,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        importance_df=importance_df,
    )

    print("\n=== Regression Results ===")
    print("Training rows: {}".format(len(X_train)))
    print("Testing rows: {}".format(len(X_test)))

    print("\nModel comparison:")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: "{:.3f}".format(value),
        )
    )

    print("\nBest model: {}".format(best_model_name))

    print("\nTop Random Forest predictors:")
    print(
        importance_df
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: "{:.4f}".format(value),
        )
    )


def main() -> None:
    """
    Execute regression when the script is run directly.
    """
    run_regression_analysis()


if __name__ == "__main__":
    main()