"""
Exploratory data analysis for the Gym Members Exercise Dataset.

This script loads the dataset, summarizes its structure, checks data quality,
reviews categorical distributions, examines numerical summaries, and creates
an initial scatter plot for session duration vs. calories burned.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/gym_members_exercise_tracking.csv")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the gym members dataset."""
    return pd.read_csv(path)


def summarize_dataset(df: pd.DataFrame) -> None:
    """Print basic structure, missing values, and summary statistics."""
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n=== Columns ===")
    for column in df.columns:
        print(f"- {column}")

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Categorical Distributions ===")
    for column in ["Workout_Type", "Gender", "Experience_Level"]:
        print(f"\n{column}:")
        print(df[column].value_counts())

    print("\n=== Numerical Summary ===")
    print(df.describe())

    print("\n=== Correlation With Calories Burned ===")
    correlations = df.corr(numeric_only=True)["Calories_Burned"].sort_values(ascending=False)
    print(correlations)


def create_scatter_plot(df: pd.DataFrame) -> None:
    """Create and save a scatter plot of session duration vs. calories burned."""
    x_col = "Session_Duration (hours)"
    y_col = "Calories_Burned"

    correlation = df[x_col].corr(df[y_col])
    print(f"\nPearson correlation between session duration and calories burned: {correlation:.3f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.6)

    plt.title("Session Duration vs. Calories Burned")
    plt.xlabel("Session Duration (Hours)")
    plt.ylabel("Calories Burned")
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURES_DIR / "session_duration_vs_calories_burned.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"\nScatter plot saved to: {output_path}")


def main() -> None:
    """Run exploratory analysis."""
    df = load_dataset(DATA_PATH)
    summarize_dataset(df)
    create_scatter_plot(df)


if __name__ == "__main__":
    main()