Bank Marketing Logistic Regression Analysis

Overview
This project analyzes the Bank Marketing dataset and builds a **Logistic Regression** model to predict whether a client will subscribe to a term deposit (`y`).

Approach
The notebook follows a complete machine learning workflow:

1. Load the dataset from Google Drive.
2. Explore the dataset structure, feature types, missing values, and class distribution.
3. Perform exploratory data analysis (EDA) using plots.
4. Preprocess the data:
   - Encode categorical features
   - Encode the target variable
   - Split the data into train and test sets
   - Standardize numeric features
5. Handle class imbalance using **class weights**.
6. Train a **Logistic Regression** classifier.
7. Evaluate the model using:
   - Accuracy
   - ROC-AUC
   - Classification report
   - Confusion matrix
8. Analyze feature importance from model coefficients.
9. Perform **5-fold stratified cross-validation**.
10. Optimize the decision threshold based on **F1-score**.
11. Save the trained model, scaler, and generated plots.

Methodology
 Dataset
- Dataset used: **Bank Marketing**
- Loaded file: `bank-full.csv`
- Shape: **45,211 rows × 17 columns**

Preprocessing
- Target column: `y`
- Target encoding:
  - `no -> 0`
  - `yes -> 1`
- Categorical variables were encoded for model training.
- Data split:
  - Training set: **36,168**
  - Test set: **9,043**
- Standard scaling was applied before Logistic Regression.

Class Imbalance
The dataset is imbalanced:
- Class 0 (No): **31,937** samples (**88.3%**)
- Class 1 (Yes): **4,231** samples (**11.7%**)
- Imbalance ratio: **7.55:1**

To address this, the model uses:
- `class_weight='balanced'`

Model
- Algorithm: **Logistic Regression**
- Penalty: `l2`
- Solver: `lbfgs`
- Max iterations: `1000`
- Random state: `42`

Findings
Test Set Performance
- **Accuracy:** 84.60%
- **ROC-AUC:** 0.9079

Classification Report
- Class `No (0)`:
  - Precision: **0.97**
  - Recall: **0.85**
  - F1-score: **0.91**
- Class `Yes (1)`:
  - Precision: **0.42**
  - Recall: **0.81**
  - F1-score: **0.55**

Cross-Validation Results
- **CV Accuracy:** 84.46% ± 0.28%
- **CV ROC-AUC:** 0.9097 ± 0.0052
- **CV F1-score:** 0.5522 ± 0.0057

Threshold Optimization
- Optimal threshold: **0.67**
- Best F1-score: **0.5844**

At the optimized threshold:
- Accuracy improved to about **88%**
- Positive class precision and recall became more balanced compared to the default threshold.

Files Produced
The notebook saves the following outputs:
- `logistic_regression_model.pkl`
- `scaler.pkl`
- `eda_plots.png`
- `model_performance.png`
- `feature_importance.png`
- `cross_validation.png`
- `threshold_optimisation.png`

Conclusion
The Logistic Regression model performed well on the Bank Marketing dataset, especially in terms of **ROC-AUC**. The use of **class balancing**, **cross-validation**, and **threshold tuning** improved reliability and made the findings more meaningful for the minority class prediction task.

