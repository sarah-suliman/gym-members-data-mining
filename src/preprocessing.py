"""
Data preprocessing pipeline for the Gym Members Exercise Dataset.

This module validates the raw dataset, removes duplicate records, checks
data quality, creates selected derived features, and saves a cleaned copy
for later clustering, classification, and regression analyses.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "gym_members_exercise_tracking.csv"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "gym_members_exercise_cleaned.csv"
)


# ---------------------------------------------------------------------------
# Expected dataset structure
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "Max_BPM",
    "Avg_BPM",
    "Resting_BPM",
    "Session_Duration (hours)",
    "Calories_Burned",
    "Workout_Type",
    "Fat_Percentage",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    "Experience_Level",
    "BMI",
]


NUMERIC_RANGE_RULES = {
    "Age": (10, 100),
    "Weight (kg)": (30, 300),
    "Height (m)": (1.0, 2.5),
    "Max_BPM": (80, 230),
    "Avg_BPM": (40, 220),
    "Resting_BPM": (30, 150),
    "Session_Duration (hours)": (0.1, 8.0),
    "Calories_Burned": (0, 5000),
    "Fat_Percentage": (1, 70),
    "Water_Intake (liters)": (0, 10),
    "Workout_Frequency (days/week)": (0, 7),
    "Experience_Level": (1, 3),
    "BMI": (10, 70),
}


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw gym member dataset.

    Parameters
    ----------
    path:
        Location of the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Confirm that the CSV is stored in the data folder."
        )

    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Confirm that all required dataset columns are present.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{missing_columns}"
        )


def summarize_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isna().sum()


def remove_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove fully duplicated rows.

    Returns
    -------
    tuple
        Cleaned DataFrame and number of duplicates removed.
    """
    duplicate_count = int(df.duplicated().sum())
    cleaned_df = df.drop_duplicates().copy()

    return cleaned_df, duplicate_count


def find_out_of_range_values(
    df: pd.DataFrame,
) -> dict[str, int]:
    """
    Count values that fall outside reasonable numerical ranges.

    These checks identify possible data-quality concerns. They do not
    automatically delete observations.
    """
    issues: dict[str, int] = {}

    for column, (minimum, maximum) in NUMERIC_RANGE_RULES.items():
        invalid_mask = ~df[column].between(
            minimum,
            maximum,
            inclusive="both",
        )

        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            issues[column] = invalid_count

    return issues


def validate_category_values(
    df: pd.DataFrame,
) -> dict[str, list[object]]:
    """
    Identify unexpected values in categorical variables.
    """
    expected_categories = {
        "Gender": {"Male", "Female"},
        "Workout_Type": {
            "Cardio",
            "HIIT",
            "Strength",
            "Yoga",
        },
        "Experience_Level": {1, 2, 3},
    }

    unexpected_values: dict[str, list[object]] = {}

    for column, allowed_values in expected_categories.items():
        observed_values = set(df[column].dropna().unique())
        invalid_values = observed_values - allowed_values

        if invalid_values:
            unexpected_values[column] = sorted(invalid_values)

    return unexpected_values


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional variables for exploratory and clustering analyses.

    Notes
    -----
    Calories_Per_Hour contains information derived from Calories_Burned.
    It should not be used as a predictor when Calories_Burned is the
    regression target because that would cause target leakage.
    """
    processed_df = df.copy()

    processed_df["Heart_Rate_Reserve"] = (
        processed_df["Max_BPM"]
        - processed_df["Resting_BPM"]
    )

    processed_df["Water_Intake_Per_Hour"] = np.where(
        processed_df["Session_Duration (hours)"] > 0,
        processed_df["Water_Intake (liters)"]
        / processed_df["Session_Duration (hours)"],
        np.nan,
    )

    processed_df["Calories_Per_Hour"] = np.where(
        processed_df["Session_Duration (hours)"] > 0,
        processed_df["Calories_Burned"]
        / processed_df["Session_Duration (hours)"],
        np.nan,
    )

    processed_df["BMI_Category"] = pd.cut(
        processed_df["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=[
            "Underweight",
            "Healthy Weight",
            "Overweight",
            "Obesity",
        ],
        right=False,
    )

    return processed_df


# ---------------------------------------------------------------------------
# Saving and reporting
# ---------------------------------------------------------------------------

def save_processed_dataset(
    df: pd.DataFrame,
    path: Path = CLEAN_DATA_PATH,
) -> None:
    """Save the cleaned dataset as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_preprocessing_report(
    raw_shape: tuple[int, int],
    processed_df: pd.DataFrame,
    missing_values: pd.Series,
    duplicate_count: int,
    range_issues: dict[str, int],
    category_issues: dict[str, list[object]],
) -> None:
    """Print a summary of preprocessing results."""
    print("\n=== Preprocessing Report ===")

    print(
        f"Original dataset: "
        f"{raw_shape[0]} rows, {raw_shape[1]} columns"
    )

    print(
        f"Processed dataset: "
        f"{processed_df.shape[0]} rows, "
        f"{processed_df.shape[1]} columns"
    )

    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Total missing values: {int(missing_values.sum())}")

    if range_issues:
        print("\nOut-of-range values detected:")
        for column, count in range_issues.items():
            print(f"- {column}: {count}")
    else:
        print("Out-of-range values detected: 0")

    if category_issues:
        print("\nUnexpected category values detected:")
        for column, values in category_issues.items():
            print(f"- {column}: {values}")
    else:
        print("Unexpected category values detected: 0")

    print("\nDerived features added:")
    print("- Heart_Rate_Reserve")
    print("- Water_Intake_Per_Hour")
    print("- Calories_Per_Hour")
    print("- BMI_Category")

    print(f"\nCleaned dataset saved to:\n{CLEAN_DATA_PATH}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset() -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Returns
    -------
    pandas.DataFrame
        Cleaned and feature-engineered dataset.
    """
    raw_df = load_dataset()
    raw_shape = raw_df.shape

    validate_required_columns(raw_df)

    missing_values = summarize_missing_values(raw_df)

    cleaned_df, duplicate_count = remove_duplicates(raw_df)

    range_issues = find_out_of_range_values(cleaned_df)
    category_issues = validate_category_values(cleaned_df)

    processed_df = add_derived_features(cleaned_df)

    save_processed_dataset(processed_df)

    print_preprocessing_report(
        raw_shape=raw_shape,
        processed_df=processed_df,
        missing_values=missing_values,
        duplicate_count=duplicate_count,
        range_issues=range_issues,
        category_issues=category_issues,
    )

    return processed_df


def main() -> None:
    """Execute preprocessing when this file is run directly."""
    preprocess_dataset()


if __name__ == "__main__":
    main()