import pandas as pd

# Load dataset
df = pd.read_csv("data/gym_members_exercise_tracking.csv")

# Basic information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nWorkout Type Counts:")
print(df["Workout_Type"].value_counts())

print("\nGender Counts:")
print(df["Gender"].value_counts())

print("\nExperience Level Counts:")
print(df["Experience_Level"].value_counts())

print("\nNumerical Summary:")
print(df.describe())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True)["Calories_Burned"].sort_values(ascending=False))

# =====================================
# Homework 2 Analysis
# =====================================

import matplotlib.pyplot as plt

# Calculate Pearson correlation
r = df["Session_Duration (hours)"].corr(df["Calories_Burned"])

print("\nHomework 2 Results")
print(f"Pearson Correlation (r): {r:.3f}")

# Create scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Session_Duration (hours)"],
    df["Calories_Burned"],
    alpha=0.6
)

plt.title("Session Duration vs Calories Burned")
plt.xlabel("Session Duration (Hours)")
plt.ylabel("Calories Burned")

plt.grid(True)

plt.tight_layout()

plt.savefig("scatter_plot.png", dpi=300)

plt.show()