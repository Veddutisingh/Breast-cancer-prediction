import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --------------------------------------------------
# Step 1: Read the CSV file
# --------------------------------------------------

data = pd.read_csv("breast_cancer.csv")

print("Breast Cancer Data (first 5 rows):")
print(data.head())
print("\nShape of data:", data.shape)

# --------------------------------------------------
# Step 2: Clean the data
#
# M (Malignant) -> 1
# B (Benign)    -> 0
# --------------------------------------------------

# id column is not useful for prediction
data = data.drop("id", axis=1)

# some versions of the file have an empty last column
if "Unnamed: 32" in data.columns:
    data = data.drop("Unnamed: 32", axis=1)

data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

print("\nMissing values:", data.isnull().sum().sum())
print("Benign patients    (0):", np.sum(data["diagnosis"] == 0))
print("Malignant patients (1):", np.sum(data["diagnosis"] == 1))

# --------------------------------------------------
# Step 3: Separate features (X) and target (Y)
# --------------------------------------------------

# Independent variables (X): all the measurements
X = data.drop("diagnosis", axis=1).values

# Dependent variable (Y): diagnosis
Y = data["diagnosis"].values

# --------------------------------------------------
# Step 4: Split into training and testing data
#
# 80% training, 20% testing
# stratify keeps the same benign/malignant ratio in both
# --------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# --------------------------------------------------
# Step 5: Scale the features
#
# X_scaled = (X - mean) / standard deviation
# mean and std are taken from TRAINING data only
# --------------------------------------------------

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X

# --------------------------------------------------
# Step 6: Train the models
#
# Model 1: Logistic Regression
# Model 2: Support Vector Machine (SVM)
# --------------------------------------------------

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, Y_train)

svm_model = SVC()
svm_model.fit(X_train_scaled, Y_train)

print("\nBoth models trained successfully")

# --------------------------------------------------
# Step 7: Predict diagnosis for test data
# --------------------------------------------------

Y_pred_lr = lr_model.predict(X_test_scaled)
Y_pred_svm = svm_model.predict(X_test_scaled)

print("\nActual and Predicted Diagnosis (first 10 patients):")
result = pd.DataFrame({
    "Actual_Diagnosis": Y_test,
    "LR_Predicted": Y_pred_lr,
    "SVM_Predicted": Y_pred_svm
})

print(result.head(10))

# --------------------------------------------------
# Step 8: Confusion matrix and evaluation metrics
#
# TP = malignant predicted as malignant
# TN = benign predicted as benign
# FP = benign predicted as malignant (false alarm)
# FN = malignant predicted as benign (missed cancer)
#
# Accuracy  = (TP + TN) / total
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
# F1 Score  = 2 * Precision * Recall / (Precision + Recall)
#
# This function is used for both models
# --------------------------------------------------

def calculate_metrics(Y_actual, Y_predicted):
    TP = np.sum((Y_actual == 1) & (Y_predicted == 1))
    TN = np.sum((Y_actual == 0) & (Y_predicted == 0))
    FP = np.sum((Y_actual == 0) & (Y_predicted == 1))
    FN = np.sum((Y_actual == 1) & (Y_predicted == 0))

    accuracy = (TP + TN) / len(Y_actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    return TP, TN, FP, FN, accuracy, precision, recall, f1_score

TP1, TN1, FP1, FN1, acc1, prec1, rec1, f1_1 = calculate_metrics(Y_test, Y_pred_lr)
TP2, TN2, FP2, FN2, acc2, prec2, rec2, f1_2 = calculate_metrics(Y_test, Y_pred_svm)

print("\nConfusion Matrix (Logistic Regression):")
print("TP =", TP1, " FP =", FP1)
print("FN =", FN1, " TN =", TN1)

print("\nConfusion Matrix (SVM):")
print("TP =", TP2, " FP =", FP2)
print("FN =", FN2, " TN =", TN2)

# --------------------------------------------------
# Step 9: Compare the two models
# --------------------------------------------------

train_acc_lr = np.mean(lr_model.predict(X_train_scaled) == Y_train)
train_acc_svm = np.mean(svm_model.predict(X_train_scaled) == Y_train)

comparison = pd.DataFrame({
    "Metric": ["Training Accuracy", "Testing Accuracy", "Precision", "Recall", "F1 Score", "Missed Cancers (FN)"],
    "Logistic Regression": [train_acc_lr, acc1, prec1, rec1, f1_1, FN1],
    "SVM": [train_acc_svm, acc2, prec2, rec2, f1_2, FN2]
})

print("\nModel Comparison:")
print(comparison)

# --------------------------------------------------
# Step 10: Predict for one patient
#
# Take the first patient from the test data
# --------------------------------------------------

patient = X_test_scaled[0].reshape(1, -1)

patient_pred_lr = lr_model.predict(patient)[0]
patient_pred_svm = svm_model.predict(patient)[0]
patient_prob = lr_model.predict_proba(patient)[0][1]

print("\nPrediction for first test patient:")
print("Logistic Regression:", "Malignant" if patient_pred_lr == 1 else "Benign",
      "(probability of cancer =", round(patient_prob, 3), ")")
print("SVM                :", "Malignant" if patient_pred_svm == 1 else "Benign")
print("Actual diagnosis   :", "Malignant" if Y_test[0] == 1 else "Benign")


# --------------------------------------------------
# Step 11: Display Confusion Matrices
# --------------------------------------------------

matrix_lr = np.array([[TN1, FP1],
                      [FN1, TP1]])

matrix_svm = np.array([[TN2, FP2],
                       [FN2, TP2]])

plt.figure(figsize=(12, 5))

# Graph 1: Logistic Regression
plt.subplot(1, 2, 1)
plt.imshow(matrix_lr, cmap="Blues")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix_lr[i, j],
                 ha="center", va="center", fontsize=16)

plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.yticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.xlabel("Predicted Diagnosis")
plt.ylabel("Actual Diagnosis")
plt.title("Logistic Regression Confusion Matrix")

# Graph 2: SVM
plt.subplot(1, 2, 2)
plt.imshow(matrix_svm, cmap="Blues")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix_svm[i, j],
                 ha="center", va="center", fontsize=16)

plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.yticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.xlabel("Predicted Diagnosis")
plt.ylabel("Actual Diagnosis")
plt.title("SVM Confusion Matrix")

plt.suptitle("Breast Cancer Classification: Confusion Matrices")
plt.tight_layout()
plt.show()
