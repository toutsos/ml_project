
# Big Data ML Project

This project implements a complete machine learning workflow for a classification task, focusing on data analysis, handling imbalanced datasets, and training multiple predictive models.

## Project Overview

The main goal is to predict admissions outcomes based on various features such as GPA, GMAT scores, race, gender, and major. The project involves a thorough data exploration phase, sophisticated preprocessing, and experimentation with multiple machine learning algorithms to evaluate and optimize predictive performance.

## Key Steps

### 1. Exploratory Data Analysis (EDA)

- Visualization of distributions (histograms, bar plots, scatter plots)
- Correlation heatmaps to identify feature relationships
- Group-based statistics (e.g., admissions by race and gender)

### 2. Data Preprocessing

- Handling missing data
- One-hot encoding of categorical features (e.g., race, major)
- Normalization of numerical data
- Feature engineering (transforming and selecting relevant features)

### 3. Addressing Class Imbalance

- **SMOTE (Synthetic Minority Oversampling Technique)** to oversample minority classes
- Downsampling of majority classes to achieve balance

### 4. Model Training and Evaluation

Implemented models:

- **Logistic Regression** (base and fine-tuned)
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Gradient Boosting Classifier**

Evaluation techniques:

- Cross-validation (with multiple metrics: accuracy, balanced accuracy, F1-score)
- Confusion matrix visualization
- ROC curve plotting
- Feature importance ranking

### 5. Hyperparameter Tuning

- Used GridSearchCV to optimize hyperparameters for Logistic Regression, Random Forest, and XGBoost
- Explored parameter ranges such as number of estimators, max depth, and regularization strength

## Results

The notebook compares different models using metrics like accuracy, F1-score, and ROC-AUC, and provides insights into the trade-offs between different balancing techniques and models. Feature importance plots highlight which factors most influence the prediction outcomes.

## How to Run

1. Install the required libraries (e.g., `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`).
2. Load the notebook in Jupyter and run all cells.
3. Modify dataset paths and parameters as needed for your environment.

## Main Techniques Used

- Data visualization
- Imbalanced data handling (SMOTE, downsampling)
- Model selection and evaluation
- Hyperparameter optimization
- Feature importance analysis

## Potential Improvements

- Extend the feature set with additional data sources
- Test deep learning models for comparison
- Implement pipelines for streamlined preprocessing and training

## License

Specify your license here (e.g., MIT License).
