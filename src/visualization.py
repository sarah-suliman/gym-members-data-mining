"""
Visualization pipeline for the Gym Members Exercise Dataset.

This script loads the cleaned dataset and generates visualizations used
to examine distributions, relationships, and differences among gym members.
All figures are saved to the project's figures directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------

FIGURE_DPI = 300

sns.set_theme(
    context="notebook",
    style="whitegrid",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned gym members dataset.

    Parameters
    ----------
    path:
        Location of the cleaned CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned gym members dataset.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {path}\n"
            "Run src/preprocessing.py before generating visualizations."
        )

    return pd.read_csv(path)


def save_figure(filename: str) -> Path:
    """
    Save the current Matplotlib figure in the figures directory.

    Parameters
    ----------
    filename:
        Name of the image file.

    Returns
    -------
    pathlib.Path
        Location where the figure was saved.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path.name}")

    return output_path


# ---------------------------------------------------------------------------
# Distribution visualizations
# ---------------------------------------------------------------------------

def plot_calories_distribution(df: pd.DataFrame) -> None:
    """Create a histogram showing the distribution of calories burned."""
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Calories_Burned",
        bins=25,
        kde=True,
    )

    plt.title("Distribution of Calories Burned")
    plt.xlabel("Calories Burned")
    plt.ylabel("Number of Gym Members")

    save_figure("calories_burned_distribution.png")


def plot_bmi_distribution(df: pd.DataFrame) -> None:
    """Create a histogram showing the distribution of BMI."""
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="BMI",
        bins=25,
        kde=True,
    )

    plt.title("Distribution of BMI")
    plt.xlabel("Body Mass Index")
    plt.ylabel("Number of Gym Members")

    save_figure("bmi_distribution.png")


def plot_workout_type_distribution(df: pd.DataFrame) -> None:
    """Create a count plot showing the distribution of workout types."""
    workout_order = (
        df["Workout_Type"]
        .value_counts()
        .index
    )

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="Workout_Type",
        order=workout_order,
    )

    plt.title("Distribution of Workout Types")
    plt.xlabel("Workout Type")
    plt.ylabel("Number of Gym Members")

    save_figure("workout_type_distribution.png")


# ---------------------------------------------------------------------------
# Relationship visualizations
# ---------------------------------------------------------------------------

def plot_session_duration_vs_calories(df: pd.DataFrame) -> None:
    """
    Plot the relationship between session duration and calories burned.

    A regression line is included to summarize the overall trend.
    """
    x_column = "Session_Duration (hours)"
    y_column = "Calories_Burned"

    correlation = df[x_column].corr(df[y_column])

    plt.figure(figsize=(10, 6))

    sns.regplot(
        data=df,
        x=x_column,
        y=y_column,
        scatter_kws={"alpha": 0.55},
        line_kws={"linewidth": 2},
    )

    plt.title(
        "Session Duration and Calories Burned\n"
        f"Pearson correlation: r = {correlation:.2f}"
    )
    plt.xlabel("Session Duration (Hours)")
    plt.ylabel("Calories Burned")

    save_figure("session_duration_vs_calories_burned.png")


def plot_calories_by_workout_type(df: pd.DataFrame) -> None:
    """Compare calories burned across workout types using box plots."""
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Workout_Type",
        y="Calories_Burned",
    )

    plt.title("Calories Burned by Workout Type")
    plt.xlabel("Workout Type")
    plt.ylabel("Calories Burned")

    save_figure("calories_burned_by_workout_type.png")


def plot_calories_by_experience_level(df: pd.DataFrame) -> None:
    """Compare calories burned across experience levels."""
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Experience_Level",
        y="Calories_Burned",
        order=[1, 2, 3],
    )

    plt.title("Calories Burned by Experience Level")
    plt.xlabel("Experience Level")
    plt.ylabel("Calories Burned")

    save_figure("calories_burned_by_experience_level.png")


# ---------------------------------------------------------------------------
# Correlation visualization
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create a heatmap of correlations among selected numerical variables."""
    selected_columns = [
        "Age",
        "BMI",
        "Fat_Percentage",
        "Resting_BPM",
        "Avg_BPM",
        "Heart_Rate_Reserve",
        "Session_Duration (hours)",
        "Workout_Frequency (days/week)",
        "Water_Intake (liters)",
        "Experience_Level",
        "Calories_Burned",
    ]

    correlation_matrix = (
        df[selected_columns]
        .corr()
    )

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
    )

    plt.title("Correlation Heatmap of Fitness and Workout Variables")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_figure("correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Visualization pipeline
# ---------------------------------------------------------------------------

def generate_all_visualizations(df: pd.DataFrame) -> None:
    """Generate and save all planned exploratory visualizations."""
    visualizations = [
        plot_calories_distribution,
        plot_bmi_distribution,
        plot_workout_type_distribution,
        plot_session_duration_vs_calories,
        plot_calories_by_workout_type,
        plot_calories_by_experience_level,
        plot_correlation_heatmap,
    ]

    print("\n=== Generating Visualizations ===")

    for visualization in visualizations:
        visualization(df)

    print(
        f"\nGenerated {len(visualizations)} visualizations."
        f"\nFigures directory: {FIGURES_DIR}"
    )


def main() -> None:
    """Run the complete visualization pipeline."""
    dataset = load_dataset()

    print(
        f"Loaded cleaned dataset: "
        f"{dataset.shape[0]} rows, {dataset.shape[1]} columns"
    )

    generate_all_visualizations(dataset)


if __name__ == "__main__":
    main()