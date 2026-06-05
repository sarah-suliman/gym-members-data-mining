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