"""
K-Means clustering analysis for the Gym Members Exercise Dataset.

This script identifies fitness profiles using physiological and workout
behavior variables. It evaluates several cluster counts, selects the best
solution using silhouette score, summarizes each cluster, and saves the
trained model, results, and visualizations.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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

CLUSTERED_DATA_PATH = RESULTS_DIR / "gym_members_with_clusters.csv"
CLUSTER_PROFILE_PATH = RESULTS_DIR / "cluster_profiles.csv"
CLUSTER_METRICS_PATH = RESULTS_DIR / "clustering_metrics.csv"

SCALER_PATH = MODELS_DIR / "clustering_scaler.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_model.joblib"


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8

CLUSTER_FEATURES = [
    "BMI",
    "Fat_Percentage",
    "Avg_BPM",
    "Resting_BPM",
    "Heart_Rate_Reserve",
    "Session_Duration (hours)",
    "Workout_Frequency (days/week)",
    "Water_Intake (liters)",
]


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
            f"Cleaned dataset not found at: {path}\n"
            "Run src/preprocessing.py before clustering."
        )

    return pd.read_csv(path)


def validate_features(df: pd.DataFrame) -> None:
    """
    Confirm that all required clustering features are available.
    """
    missing_features = [
        feature
        for feature in CLUSTER_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The dataset is missing clustering features: "
            f"{missing_features}"
        )

    missing_values = int(df[CLUSTER_FEATURES].isna().sum().sum())

    if missing_values > 0:
        raise ValueError(
            "Clustering features contain missing values. "
            f"Missing-value count: {missing_values}"
        )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_clustering_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Select and standardize the clustering variables.

    Standardization is necessary because K-Means uses Euclidean distance.
    Variables measured on larger scales would otherwise dominate the model.
    """
    feature_data = df[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(feature_data)

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=CLUSTER_FEATURES,
        index=df.index,
    )

    return scaled_df, scaler


# ---------------------------------------------------------------------------
# Cluster-count evaluation
# ---------------------------------------------------------------------------

def evaluate_cluster_counts(
    scaled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit K-Means models across several cluster counts.

    Returns
    -------
    pandas.DataFrame
        Table containing SSE and silhouette score for each value of k.
    """
    evaluation_rows = []

    for cluster_count in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(scaled_df)

        evaluation_rows.append(
            {
                "k": cluster_count,
                "SSE": model.inertia_,
                "Silhouette_Score": silhouette_score(
                    scaled_df,
                    labels,
                ),
            }
        )

    return pd.DataFrame(evaluation_rows)


def select_best_cluster_count(
    metrics_df: pd.DataFrame,
) -> int:
    """
    Select the value of k with the highest silhouette score.
    """
    best_row = metrics_df.loc[
        metrics_df["Silhouette_Score"].idxmax()
    ]

    return int(best_row["k"])


# ---------------------------------------------------------------------------
# Final model and profiles
# ---------------------------------------------------------------------------

def fit_final_model(
    scaled_df: pd.DataFrame,
    cluster_count: int,
) -> tuple[KMeans, pd.Series]:
    """
    Fit the final K-Means model and return cluster labels.
    """
    model = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(scaled_df)

    label_series = pd.Series(
        labels,
        index=scaled_df.index,
        name="Cluster",
    )

    return model, label_series


def create_cluster_profiles(
    clustered_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary statistics describing each cluster.
    """
    profile_columns = CLUSTER_FEATURES + [
        "Calories_Burned",
        "Experience_Level",
    ]

    profiles = (
        clustered_df
        .groupby("Cluster")[profile_columns]
        .mean()
        .round(2)
    )

    cluster_sizes = (
        clustered_df["Cluster"]
        .value_counts()
        .sort_index()
        .rename("Cluster_Size")
    )

    profiles.insert(
        0,
        "Cluster_Size",
        cluster_sizes,
    )

    return profiles


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_current_figure(filename: str) -> Path:
    """
    Save and close the current Matplotlib figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved figure: {output_path.name}")

    return output_path


def plot_elbow_curve(metrics_df: pd.DataFrame) -> None:
    """
    Plot SSE across candidate cluster counts.
    """
    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=metrics_df,
        x="k",
        y="SSE",
        marker="o",
    )

    plt.title("K-Means Elbow Curve")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster SSE")
    plt.xticks(metrics_df["k"])

    save_current_figure("kmeans_elbow_curve.png")


def plot_silhouette_scores(
    metrics_df: pd.DataFrame,
    best_k: int,
) -> None:
    """
    Plot silhouette scores for each candidate cluster count.
    """
    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=metrics_df,
        x="k",
        y="Silhouette_Score",
        marker="o",
    )

    best_score = metrics_df.loc[
        metrics_df["k"] == best_k,
        "Silhouette_Score",
    ].iloc[0]

    plt.scatter(
        best_k,
        best_score,
        s=120,
        label=f"Selected k = {best_k}",
        zorder=5,
    )

    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(metrics_df["k"])
    plt.legend()

    save_current_figure("kmeans_silhouette_scores.png")


def create_pca_projection(
    scaled_df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """
    Reduce the standardized features to two PCA components for plotting.

    PCA is used only for visualization. The K-Means model is trained using
    all clustering features.
    """
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    components = pca.fit_transform(scaled_df)

    return pd.DataFrame(
        {
            "Principal_Component_1": components[:, 0],
            "Principal_Component_2": components[:, 1],
            "Cluster": cluster_labels.astype(str),
        },
        index=scaled_df.index,
    )


def plot_clusters_pca(pca_df: pd.DataFrame) -> None:
    """
    Display the clusters in a two-dimensional PCA projection.
    """
    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=pca_df,
        x="Principal_Component_1",
        y="Principal_Component_2",
        hue="Cluster",
        alpha=0.7,
        s=55,
    )

    plt.title("Gym Member Fitness Profiles Using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")

    save_current_figure("kmeans_clusters_pca.png")


def plot_cluster_profiles(
    profiles_df: pd.DataFrame,
) -> None:
    """
    Create a standardized heatmap comparing cluster profiles.
    """
    profile_features = CLUSTER_FEATURES + [
        "Calories_Burned",
        "Experience_Level",
    ]

    profile_values = profiles_df[profile_features]

    standardized_profiles = (
        profile_values - profile_values.mean()
    ) / profile_values.std(ddof=0)

    plt.figure(figsize=(14, 6))

    sns.heatmap(
        standardized_profiles,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        center=0,
        cbar_kws={
            "label": "Standardized Cluster Mean"
        },
    )

    plt.title("Standardized Fitness Profile Characteristics")
    plt.xlabel("Fitness and Workout Variable")
    plt.ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_current_figure("cluster_profile_heatmap.png")


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------

def save_results(
    clustered_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    scaler: StandardScaler,
    model: KMeans,
) -> None:
    """
    Save datasets, metrics, and fitted preprocessing/model objects.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    clustered_df.to_csv(
        CLUSTERED_DATA_PATH,
        index=False,
    )

    profiles_df.to_csv(CLUSTER_PROFILE_PATH)

    metrics_df.to_csv(
        CLUSTER_METRICS_PATH,
        index=False,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    joblib.dump(
        model,
        KMEANS_MODEL_PATH,
    )

    print(f"Saved clustered data: {CLUSTERED_DATA_PATH.name}")
    print(f"Saved cluster profiles: {CLUSTER_PROFILE_PATH.name}")
    print(f"Saved clustering metrics: {CLUSTER_METRICS_PATH.name}")
    print(f"Saved scaler: {SCALER_PATH.name}")
    print(f"Saved K-Means model: {KMEANS_MODEL_PATH.name}")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_clustering_analysis() -> None:
    """
    Run the complete K-Means clustering workflow.
    """
    df = load_dataset()
    validate_features(df)

    scaled_df, scaler = prepare_clustering_features(df)

    metrics_df = evaluate_cluster_counts(scaled_df)
    best_k = select_best_cluster_count(metrics_df)

    model, cluster_labels = fit_final_model(
        scaled_df,
        best_k,
    )

    clustered_df = df.copy()
    clustered_df["Cluster"] = cluster_labels

    profiles_df = create_cluster_profiles(clustered_df)

    pca_df = create_pca_projection(
        scaled_df,
        cluster_labels,
    )

    plot_elbow_curve(metrics_df)
    plot_silhouette_scores(metrics_df, best_k)
    plot_clusters_pca(pca_df)
    plot_cluster_profiles(profiles_df)

    save_results(
        clustered_df=clustered_df,
        profiles_df=profiles_df,
        metrics_df=metrics_df,
        scaler=scaler,
        model=model,
    )

    selected_score = metrics_df.loc[
        metrics_df["k"] == best_k,
        "Silhouette_Score",
    ].iloc[0]

    print("\n=== Clustering Results ===")
    print(f"Features used: {len(CLUSTER_FEATURES)}")
    print(f"Selected number of clusters: {best_k}")
    print(f"Silhouette score: {selected_score:.3f}")

    print("\nCluster sizes:")
    print(
        clustered_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster profiles:")
    print(profiles_df)


def main() -> None:
    """Execute clustering when the script is run directly."""
    run_clustering_analysis()


if __name__ == "__main__":
    main()