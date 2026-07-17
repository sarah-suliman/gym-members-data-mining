"""
Classification analysis for the Gym Members Exercise Dataset.

This script predicts gym member experience level using demographic,
physiological, and workout-related variables. It compares a majority-class
baseline, Decision Tree, and Random Forest classifier, then saves evaluation
metrics, predictions, trained models, and visualizations.

Compatible with Python 3.9.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

METRICS_PATH = RESULTS_DIR / "classification_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "classification_predictions.csv"
REPORT_PATH = RESULTS_DIR / "classification_report.csv"

BASELINE_PATH = MODELS_DIR / "baseline_classifier.joblib"
DECISION_TREE_PATH = MODELS_DIR / "decision_tree_classifier.joblib"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest_classifier.joblib"


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Experience_Level"

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
    "Calories_Burned",
    "Fat_Percentage",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
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

    Parameters
    ----------
    path:
        Path to the cleaned CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned gym member dataset.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset cannot be found.
    """
    if not path.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found at: {}\n"
            "Run src/preprocessing.py before classification.".format(path)
        )

    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Confirm that all required features and the target are present.

    Parameters
    ----------
    df:
        Dataset to validate.

    Raises
    ------
    ValueError
        If required columns are missing or contain missing values.
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
            "Classification data contains missing values. "
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
    Split data into stratified training and test sets.

    Stratification preserves the proportion of each experience level.

    Parameters
    ----------
    df:
        Cleaned gym member dataset.

    Returns
    -------
    tuple
        X_train, X_test, y_train, and y_test.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing steps for numeric and categorical variables.

    Numerical variables are standardized. Categorical variables are
    one-hot encoded.
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
    Build the baseline and machine learning classification pipelines.

    Returns
    -------
    dict
        Model names mapped to scikit-learn pipelines.
    """
    baseline_model = Pipeline(
        steps=[
            (
                "classifier",
                DummyClassifier(
                    strategy="most_frequent",
                ),
            ),
        ]
    )

    decision_tree_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=8,
                    random_state=RANDOM_STATE,
                ),
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
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Baseline": baseline_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
    Fit and evaluate one classification model.

    Parameters
    ----------
    model_name:
        Human-readable model name.
    model:
        Scikit-learn pipeline.
    X_train:
        Training features.
    X_test:
        Test features.
    y_train:
        Training target.
    y_test:
        Test target.

    Returns
    -------
    tuple
        Metric dictionary and prediction Series.
    """
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision_Weighted": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Recall_Weighted": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1_Weighted": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1_Macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    prediction_series = pd.Series(
        predictions,
        index=y_test.index,
        name=model_name,
    )

    return metrics, prediction_series


def create_classification_report(
    y_test: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """
    Create class-level precision, recall, and F1 statistics.
    """
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_current_figure(filename: str) -> Path:
    """
    Save and close the current Matplotlib figure.

    Parameters
    ----------
    filename:
        Name of the output image.

    Returns
    -------
    pathlib.Path
        Saved figure path.
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
    Compare classification models using weighted F1-score.
    """
    ordered_metrics = metrics_df.sort_values(
        "F1_Weighted",
        ascending=False,
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=ordered_metrics,
        x="Model",
        y="F1_Weighted",
    )

    plt.title("Classification Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Weighted F1-Score")
    plt.ylim(0, 1)

    save_current_figure(
        "classification_model_comparison.png"
    )


def plot_confusion_matrix(
    y_test: pd.Series,
    predictions: pd.Series,
    model_name: str,
) -> None:
    """
    Create a confusion matrix for the selected model.
    """
    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )

    plt.title("{} Confusion Matrix".format(model_name))
    plt.xlabel("Predicted Experience Level")
    plt.ylabel("Actual Experience Level")

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    save_current_figure(
        "{}_confusion_matrix.png".format(safe_name)
    )


def plot_random_forest_feature_importance(
    model: Pipeline,
) -> None:
    """
    Plot the most influential Random Forest features.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": classifier.feature_importances_,
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

    top_features = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .head(12)
        .sort_values("Importance")
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
    )

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    save_current_figure(
        "random_forest_feature_importance.png"
    )


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_results(
    models: Dict[str, Pipeline],
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    report_df: pd.DataFrame,
) -> None:
    """
    Save models, metrics, predictions, and classification report.
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

    report_df.to_csv(REPORT_PATH)

    joblib.dump(
        models["Baseline"],
        BASELINE_PATH,
    )

    joblib.dump(
        models["Decision Tree"],
        DECISION_TREE_PATH,
    )

    joblib.dump(
        models["Random Forest"],
        RANDOM_FOREST_PATH,
    )

    print("Saved metrics: {}".format(METRICS_PATH.name))
    print("Saved predictions: {}".format(PREDICTIONS_PATH.name))
    print("Saved report: {}".format(REPORT_PATH.name))
    print("Saved model: {}".format(BASELINE_PATH.name))
    print("Saved model: {}".format(DECISION_TREE_PATH.name))
    print("Saved model: {}".format(RANDOM_FOREST_PATH.name))


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_classification_analysis() -> None:
    """
    Run the complete experience-level classification workflow.
    """
    df = load_dataset()
    validate_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(df)

    models = build_models()

    metric_rows = []
    prediction_columns = {
        "Actual_Experience_Level": y_test,
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
            "F1_Weighted",
            ascending=False,
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

    report_df = create_classification_report(
        y_test,
        best_predictions,
    )

    plot_model_comparison(metrics_df)

    plot_confusion_matrix(
        y_test,
        best_predictions,
        best_model_name,
    )

    plot_random_forest_feature_importance(
        fitted_models["Random Forest"]
    )

    save_results(
        models=fitted_models,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        report_df=report_df,
    )

    print("\n=== Classification Results ===")
    print("Training rows: {}".format(len(X_train)))
    print("Testing rows: {}".format(len(X_test)))

    print("\nClass distribution in test data:")
    print(
        y_test
        .value_counts()
        .sort_index()
    )

    print("\nModel comparison:")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: "{:.3f}".format(value),
        )
    )

    print("\nBest model: {}".format(best_model_name))

    print("\nBest model classification report:")
    print(report_df)


def main() -> None:
    """
    Execute classification when this file is run directly.
    """
    run_classification_analysis()


if __name__ == "__main__":
    main()