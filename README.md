# Credit Card Fraud Detection using Machine Learning

This project builds machine learning models to detect fraudulent credit card transactions using a real-world Kaggle dataset (284,807 transactions, only 492 fraud).

## Objective
Detect fraud as accurately as possible on a highly imbalanced dataset.

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest)
- Matplotlib, Seaborn

## Approach
1. Loaded the public *Credit Card Fraud Detection* dataset from Kaggle.
2. Performed basic exploration and visualized class imbalance (fraud vs non-fraud).
3. Scaled the **Time** and **Amount** features with `StandardScaler`.
4. Split the data into train and test sets with stratified sampling.
5. Trained and evaluated two models:
   - Logistic Regression (class_weight="balanced")
   - Random Forest (class_weight="balanced")
6. Compared models using classification report, confusion matrix, and ROC-AUC.

## Results (example)
- Logistic Regression ROC-AUC: ~0.97  
- Random Forest ROC-AUC: ~0.98–0.99  

Random Forest performed best on this dataset.

## Future Work
- Try SMOTE or other oversampling techniques
- Tune Random Forest hyperparameters
- Deploy the model as a simple API or web app
