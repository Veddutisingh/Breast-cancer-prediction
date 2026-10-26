import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
# Step 6: Train the Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

print("\nModel trained successfully")

# --------------------------------------------------
# Step 7: Predict diagnosis for test data
# --------------------------------------------------

Y_pred = model.predict(X_test_scaled)

print("\nActual and Predicted Diagnosis (first 10 patients):")
result = pd.DataFrame({
    "Actual_Diagnosis": Y_test,
    "Predicted_Diagnosis": Y_pred
})

print(result.head(10))

# --------------------------------------------------
# Step 8: Confusion matrix
#
# TP = malignant predicted as malignant
# TN = benign predicted as benign
# FP = benign predicted as malignant (false alarm)
# FN = malignant predicted as benign (missed cancer)
# --------------------------------------------------

TP = np.sum((Y_test == 1) & (Y_pred == 1))
TN = np.sum((Y_test == 0) & (Y_pred == 0))
FP = np.sum((Y_test == 0) & (Y_pred == 1))
FN = np.sum((Y_test == 1) & (Y_pred == 0))
# --------------------------------------------------
# Step 9: Calculate evaluation metrics
#
# Accuracy  = (TP + TN) / total
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
# F1 Score  = 2 * Precision * Recall / (Precision + Recall)
# --------------------------------------------------

accuracy = (TP + TN) / len(Y_test)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

cm = np.array([[TN, FP],
               [FN, TP]])

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
plt.yticks([0, 1], ["Benign (0)", "Malignant (1)"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

train_accuracy = np.mean(model.predict(X_train_scaled) == Y_train)

print("\nEvaluation Metrics:")
print("Training Accuracy :", train_accuracy)
print("Testing Accuracy  :", accuracy)
print("Precision         :", precision)
print("Recall            :", recall)
print("F1 Score          :", f1_score)

# --------------------------------------------------
# Step 10: Predict for one patient
#
# Take the first patient from the test data
# --------------------------------------------------

patient = X_test_scaled[0].reshape(1, -1)

patient_pred = model.predict(patient)[0]
patient_prob = model.predict_proba(patient)[0][1]

print("\nPrediction for first test patient:")
if patient_pred == 1:
    print("Malignant (probability =", round(patient_prob, 3), ")")
else:
    print("Benign (probability of cancer =", round(patient_prob, 3), ")")

print("Actual diagnosis:", "Malignant" if Y_test[0] == 1 else "Benign")

# --------------------------------------------------
# Step 11: Draw the graph (Confusion Matrix)
# --------------------------------------------------

matrix = np.array([[TN, FP],
                   [FN, TP]])

plt.figure(figsize=(6, 5))

plt.imshow(matrix, cmap='Blues')

# Write the numbers inside the boxes
for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix[i, j], ha='center', va='center', fontsize=20)

# Labels and title
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.yticks([0, 1], ["Benign", "Malignant"])

plt.xlabel("Predicted Diagnosis")
plt.ylabel("Actual Diagnosis")

plt.title("Confusion Matrix: Breast Cancer Prediction")

# Display graph
plt.show()
