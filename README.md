# Gym Members Data Mining Project

## Overview

This project was completed for **CSPB 4502: Data Mining** at the **University of Colorado Boulder**.

The goal of this project is to apply data mining techniques to a real-world fitness dataset to discover meaningful exercise patterns, identify groups of gym members with similar characteristics, predict workout experience level, and estimate calories burned during exercise sessions.

The project demonstrates a complete machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, clustering, classification, regression, visualization, and model evaluation.

---

# Objectives

The project addresses the following research questions:

1. Can gym members be grouped into meaningful fitness profiles using clustering?
2. Can a member's workout experience level be predicted using demographic and physiological characteristics?
3. Can calories burned during a workout be accurately predicted using measurable fitness variables?

---

# Dataset

**Dataset:** Gym Members Exercise Tracking Dataset

The dataset contains:

- **973 gym members**
- **15 original variables**
- **19 variables after preprocessing and feature engineering**

The dataset includes information such as:

- Age
- Gender
- Weight
- Height
- Heart Rate
- Workout Type
- Session Duration
- Water Intake
- Workout Frequency
- Experience Level
- BMI
- Calories Burned

During preprocessing, four additional features were created:

- Heart Rate Reserve
- Calories Per Hour
- Water Intake Per Hour
- BMI Category

---

# Repository Structure

```
gym-members-project/
│
├── data/
│   ├── processed/
│   └── results/
│
├── figures/
│
├── models/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_clustering_analysis.ipynb
│   ├── 04_classification_models.ipynb
│   ├── 05_regression_analysis.ipynb
│   └── 06_final_results.ipynb
│
├── src/
│   ├── explore_data.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── clustering.py
│   ├── classification.py
│   └── regression.py
│
├── README.md
└── requirements.txt
```

---

# Project Workflow

The project follows a complete data mining pipeline:

1. Exploratory Data Analysis
2. Data Cleaning
3. Feature Engineering
4. Data Visualization
5. K-Means Clustering
6. Classification
7. Regression
8. Model Evaluation
9. Final Results

---

# Machine Learning Techniques

## Exploratory Data Analysis

- Summary statistics
- Correlation analysis
- Distribution analysis
- Scatter plots
- Heatmaps

---

## Data Preprocessing

- Missing value detection
- Duplicate detection
- Category validation
- Range validation
- Feature engineering
- Processed dataset generation

---

## Clustering

Algorithm:

- K-Means Clustering

Evaluation:

- Elbow Method
- Silhouette Score
- PCA Visualization

---

## Classification

Models:

- Baseline Classifier
- Decision Tree Classifier
- Random Forest Classifier

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Regression

Models:

- Baseline Regressor
- Linear Regression
- Random Forest Regression

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Results

## Exploratory Analysis

The strongest relationship observed in the dataset was between workout session duration and calories burned.

- Pearson Correlation = **0.908**

Longer workout sessions were strongly associated with greater calorie expenditure.

---

## Clustering

K-Means clustering identified distinct fitness profiles among gym members.

Performance:

- Silhouette Score ≈ **0.271**

Although some overlap exists between clusters, meaningful groups of gym members were identified.

---

## Classification

Classification models successfully predicted workout experience level using demographic, physiological, and workout-related variables.

The Random Forest classifier achieved the strongest overall classification performance among the evaluated models.

---

## Regression

Linear Regression produced the strongest predictive performance for estimating calories burned.

Performance:

- MAE ≈ **30.27**
- RMSE ≈ **40.57**
- R² ≈ **0.980**

The regression model explained approximately **98%** of the variation in calories burned within the dataset.

---

# Visualizations

The project automatically generates a variety of figures, including:

- Correlation Heatmap
- Session Duration vs Calories Burned
- Workout Type Distribution
- BMI Distribution
- Calories Burned Distribution
- Calories Burned by Workout Type
- Calories Burned by Experience Level
- K-Means Elbow Curve
- PCA Cluster Visualization
- Silhouette Score Plot
- Classification Model Comparison
- Decision Tree Confusion Matrix
- Regression Model Comparison
- Feature Importance Plots
- Actual vs Predicted Calories
- Residual Plot

All figures are saved automatically to the **figures/** directory.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sarah-suliman/gym-members-data-mining.git
```

Navigate to the project:

```bash
cd gym-members-data-mining
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the scripts individually:

```bash
python src/explore_data.py
python src/preprocessing.py
python src/visualization.py
python src/clustering.py
python src/classification.py
python src/regression.py
```

The scripts automatically generate:

- Processed datasets
- Figures
- Saved machine learning models
- Evaluation metrics
- Prediction files

---

# Future Work

Potential improvements include:

- Collecting a larger and more diverse fitness dataset
- Incorporating wearable device data
- Performing hyperparameter optimization
- Comparing additional clustering algorithms
- Evaluating gradient boosting and neural network models
- Deploying the regression model as a web application

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

# References

Han, Jiawei, Micheline Kamber, and Jian Pei. *Data Mining: Concepts and Techniques*. 3rd ed., Morgan Kaufmann, 2012.

Gym Members Exercise Tracking Dataset. Kaggle.

Scikit-learn Developers. *Scikit-learn Documentation*. https://scikit-learn.org/

Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, vol. 12, 2011, pp. 2825–2830.

---

# Author

**Sarah Suliman**

Computer Science, University of Colorado Boulder

Course: CSPB 4502 – Data Mining