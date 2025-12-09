import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("creditcard.csv")

print("Data loaded successfully")
print(data.head())

# Split features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Scale 'Time' and 'Amount'
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 1) Logistic Regression
# =========================
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Results ===")
print(classification_report(y_test, y_pred_lr, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

# =========================
# 2) Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Results ===")
print(classification_report(y_test, y_pred_rf, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

# Simple comparison summary
print("\n=== Model Comparison (ROC-AUC) ===")
print("Logistic Regression:", roc_auc_score(y_test, y_proba_lr))
print("Random Forest      :", roc_auc_score(y_test, y_proba_rf))


plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Non-Fraud Transaction Count")
plt.show()
