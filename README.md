# 🏆 INFORMS Data Challenge Solution

This repository contains the complete attempt and final solution for the **INFORMS Data Challenge**. It includes all phases of the data science lifecycle, from initial data exploration and preprocessing to advanced model training, evaluation, and documentation.

---

## 🌟 Overview

The primary goal of this project was to tackle the challenge posed by INFORMS, leveraging quantitative methods and machine learning models to solve a specific operations research or data-driven business problem (e.g., forecasting, optimization, classification, or regression).

This solution demonstrates proficiency in:
* **Data Wrangling and Feature Engineering**
* **Exploratory Data Analysis (EDA)** and visualization.
* **Advanced Predictive Modeling** using ensemble techniques.
* **Performance Evaluation** and reporting.

---

## 📂 Repository Structure

The project is organized into logical folders mirroring a standard data science workflow.

| Directory | Purpose | Key Contents |
| :--- | :--- | :--- |
| **`Data Given for Challenge/`** | Stores the **raw, original input data** provided for the challenge. | Original data files (e.g., `.csv`, `.xlsx`). |
| **`Data Exploration/`** | Notebooks dedicated to **Exploratory Data Analysis (EDA)** and initial data assessment. | Jupyter notebooks (`.ipynb`) with visualizations and summaries. |
| **`Data Prep and Model Training/`** | Contains scripts and notebooks for **data cleaning, feature engineering, and model development.** | Python scripts (`.py`), training notebooks, preprocessing functions. |
| **`Documents/`** | Stores final reports, presentations, and any written summaries submitted for the challenge. | Final report (`.pdf`), presentation slides (`.pptx`). |
| **`results/`** | Holds the final outputs, including model predictions, performance metrics, and key visualization plots. | Prediction files, metric reports, model output plots. |
| **`Sample Code/code/`** | Auxiliary or utility scripts and functions used throughout the project. | Helper functions, reusable code snippets. |

---

## 📈 Data Exploration Based on Files

The `Data Exploration/` folder is crucial for understanding the raw data and informing the modeling strategy.

### **1. Data Loading and Quality Check**
* **Objective:** Load data from `Data Given for Challenge/`.
* **Focus:** Checking data types, identifying the dimensions (rows and columns), and assessing data quality.
* **Deliverables:** Notebooks showing a high-level `df.info()` and `df.describe()` to summarize numerical features.

### **2. Missing Values and Outliers**
* **Objective:** Identify the extent of missing data (`NaNs`) and the presence of extreme outliers.
* **Focus:** Visualizing missing patterns (e.g., using a heatmap or percentage bar plot) and using box plots/histograms to spot outliers in key variables.

### **3. Univariate & Bivariate Analysis**
* **Objective:** Understand the distribution of individual features and their relationship with the target variable.
* **Focus:**
    * **Target Variable Analysis:** Determining if the problem is classification (imbalanced classes?) or regression (distribution?).
    * **Feature Distributions:** Histograms and KDE plots for numerical features, and bar charts for categorical features.
    * **Feature Correlation:** Heatmaps to visualize correlations between features and the target.

### **4. Key Discoveries**
* The EDA phase culminates in a summary of key findings, such as which features are most correlated with the target, what transformation might be needed (e.g., log transformation for skewed data), and how missing values should be handled (imputation strategy).

---

## 🧠 Data Preparation and Model Training

The `Data Prep and Model Training/` folder contains the core machinery of the solution.

### **1. Data Preprocessing**
The scripts in this section execute the following steps:
* **Handling Missing Values:** Implementing specific imputation strategies (e.g., mean/median imputation, or using a specific value for categorical data) based on EDA findings.
* **Feature Engineering:** Creating new features to enhance model performance. This may include:
    * Time-series features (if applicable): Lagged values, rolling means, date/time components.
    * Interaction features: Combining two or more existing features.
* **Categorical Encoding:** Converting categorical variables into a numerical format suitable for machine learning models (e.g., One-Hot Encoding or Target Encoding).
* **Feature Scaling:** Standardizing or normalizing numerical features to ensure all features contribute equally to the distance metrics during training.

### **2. Model Selection and Training**
Given the complexity of INFORMS data challenges, **Ensemble Tree-based Models** are typically employed for their high performance and robustness.

| Model | Technique | Key Advantages |
| :--- | :--- | :--- |
| **XGBoost** | Gradient Boosting Machines | High accuracy, parallel processing, effective regularization. |
| **LightGBM** | Gradient Boosting Machines | Faster training speed, lower memory usage, specialized for large datasets. |
| **Scikit-learn (e.g., Random Forest)** | Bagging or other common techniques | Baseline model for performance comparison. |

### **3. Hyperparameter Optimization**
* **Method:** **Grid Search** or **Randomized Search** are used to efficiently find the optimal combination of model hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`).
* **Validation:** **Cross-validation** (e.g., K-Fold or Stratified K-Fold) is used throughout the training process to ensure the model generalizes well and prevents overfitting.

### **4. Model Evaluation**
* The performance of the final selected model is evaluated using appropriate metrics (e.g., **F1-Score, AUC** for classification; **RMSE, MAE** for regression) on a held-out test set.
* **Feature Importance:** Model interpretability is explored by analyzing feature importance scores to validate initial hypotheses from the EDA.

---

## 🙋‍♂️🙋‍♂️🙋‍♀️ Contributors  
- **Shreyas Rajapur** ([@Shreyasrs23](https://github.com/Shreyasrs23)) 
- **Kumar Mantha** ([@cosmicbeing619](https://github.com/cosmicbeing619)) 
- **Anuva Negi** ([@silverfrost702](https://github.com/silverfrost702))  


## 🚀 Getting Started

To run this project locally, follow these steps.

### **1. Clone the Repository**
```bash
git clone [https://github.com/Shreyasrs23/INFORMS---Data-Challenge.git](https://github.com/Shreyasrs23/INFORMS---Data-Challenge.git)
cd INFORMS---Data-Challenge

