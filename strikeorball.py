#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Load data from CSV file
data = pd.read_csv("sample-pitch-dat.csv")

# Preprocess, clean, delete rows with more than 10 consecutive NULL values
num_rows_before = data.shape[0]
data = data.loc[(data.isnull().sum(axis=1) < 10)]
num_rows_after = data.shape[0]
num_rows_deleted = num_rows_before - num_rows_after
print(f"Deleted {num_rows_deleted} rows with more than 10 consecutive NULL values.")
print(f"{num_rows_after} rows remain.")

# Clean, impute null values with column mean
data = data.fillna(data.mean())
data["pitchtypeid"] = data["pitchtypeid"].astype(int) 

# Define predictor variables
predictors = ["IsLefthandedPitcher", "IsLefthandedBatter", "inning", "balls", "strikes", "pitchtypeid", "ReleaseSpeed", "PlateX", "PlateZ", "ReleaseX", "ReleaseZ", "Extension", "SpinRate", "SpinDirection", "X0", "Y0", "Z0", "VerticalBreak", "InducedVertBreak"]


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data["isstrike"], test_size=0.3, random_state=42)

# Train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# Define a function to generate a random value within the range of a column
def generate_random_value(column):
    if column.dtype == int:
        return column.sample(1).iloc[0]
    else:
        min_value = column.min()
        max_value = column.max()
        return np.random.uniform(min_value, max_value)

# Get user input for predictor values
custom_input = pd.DataFrame(columns=predictors, index=[0])
for predictor in predictors:
    input_value = input(f"Enter a value for {predictor} (type 'r' for random value within column range, or leave blank for median): ")
    if input_value == 'r':
        custom_input.at[0, predictor] = generate_random_value(X_train[predictor])
    elif input_value:
        custom_input.at[0, predictor] = float(input_value)
    else:
        custom_input.at[0, predictor] = X_train[predictor].median()


# Predict on test set
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)

# Precision, Recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for 'strikeorball'")
plt.savefig("confusion_matrix.png")

# Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, data[predictors], data["isstrike"], cv=cv, scoring="accuracy")
print("\nCross-Validation Scores (Accuracy):", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\nROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for "strikeorball"')
plt.legend(loc="lower right")
plt.savefig("strikeorball_roc_curve.png")

# Print feature importances
importances = rf.feature_importances_
importance_ranking = sorted(zip(importances, predictors), reverse=True)

print("\nFeature Importances:")
for importance, feature in importance_ranking:
    print(f"{feature}: {importance:.4f}")

plt.show()

custom_input_pred_prob = rf.predict_proba(custom_input)[:, 1]

print("The predicted probability of a striiiiiiike being called:", custom_input_pred_prob[0])
print("The predicted probability of a ball being called:", 1 - custom_input_pred_prob[0])

# Ask if user wants to see the values used for each predictor variable
show_input_values = input("Do you want to see the values used for each predictor variable in this most recent run? (y/n): ")
if show_input_values.lower() == "y":
    print("\nPredictor variable values:")
    for predictor in predictors:
        print(f"{predictor}: {custom_input.at[0, predictor]}")
else:
    print("\nExiting...")