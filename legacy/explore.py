import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import pickle

# Load dataset
df = pd.read_csv("Autism_Dataset_Combined.csv")
df.drop_duplicates(inplace=True)

# Label Encoding (for binary categorical columns)
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    if df[col].nunique() == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].str.lower())
        label_encoders[col] = le

# One-Hot Encoding for categorical variables with >2 classes
df = pd.get_dummies(df, columns=[col for col in categorical_cols if df[col].nunique() > 2], drop_first=True)

# Handle numerical data and remove outliers
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
z_scores = np.abs((df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std())
df = df[(z_scores < 3).all(axis=1)]

# Correlation Matrix Plot for Features
corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, annot_kws={"size": 8})
plt.title('Correlation Matrix of ASD Features', fontsize=14)
plt.savefig('correlation_matrix_features.png')

# Feature Importance with 3D Plot
X_features = df.drop("Class", axis=1)
y_target = df["Class"]

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_features, y_target)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features
top_features = X_features.columns[indices]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x_pos = np.arange(len(top_features))
y_pos = np.zeros_like(x_pos)
z_pos = np.zeros_like(x_pos)
dx = dy = np.ones_like(x_pos)
dz = importances[indices]

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='teal')
ax.set_xticks(x_pos)
ax.set_xticklabels(top_features, rotation=90)
ax.set_title("Top 20 Feature Importances (Random Forest)")
ax.set_zlabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance_3d.png')

# Reassign for next phase
df = pd.read_csv("cleaned_dataset.csv")  # Reload cleaned dataset if required
X = df.drop("Class", axis=1)
y = df["Class"]

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTraining Set Size: {X_train.shape}, Testing Set Size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
numeric_features = ['Age']
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Grid Search Setup
param_grid_svm = {'C': [0.1], 'kernel': ['linear']}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
param_grid_dt = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'gamma': [0, 0.1],
    'subsample': [0.7, 1.0]
}

# Model training
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=5)
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
xgb_grid = GridSearchCV(XGBClassifier(random_state=42), param_grid_xgb, cv=5)

svm_grid.fit(X_train_scaled, y_train)
knn_grid.fit(X_train_scaled, y_train)
dt_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)

# Best models
best_svm = svm_grid.best_estimator_
best_knn = knn_grid.best_estimator_
best_dt = dt_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

# Cross-validated predictions (stacking base)
svm_pred = cross_val_predict(best_svm, X_train_scaled, y_train, cv=5, method="predict_proba")[:, 1]
knn_pred = cross_val_predict(best_knn, X_train_scaled, y_train, cv=5, method="predict_proba")[:, 1]
dt_pred = cross_val_predict(best_dt, X_train, y_train, cv=5, method="predict_proba")[:, 1]

# Stacking
stacked_train = pd.DataFrame({'SVM': svm_pred, 'KNN': knn_pred, 'DecisionTree': dt_pred})
best_xgb.fit(stacked_train, y_train)

# Test predictions
svm_test = best_svm.predict(X_test_scaled)
knn_test = best_knn.predict(X_test_scaled)
dt_test = best_dt.predict(X_test)
svm_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
knn_proba = best_knn.predict_proba(X_test_scaled)[:, 1]
dt_proba = best_dt.predict_proba(X_test)[:, 1]

# Stack test set for final prediction
stacked_test = pd.DataFrame({'SVM': svm_proba, 'KNN': knn_proba, 'DecisionTree': dt_proba})
stacked_pred = best_xgb.predict(stacked_test)

# 🔍 Correlation Matrix of Model Predictions
model_predictions = pd.DataFrame({
    'SVM': svm_proba,
    'KNN': knn_proba,
    'DecisionTree': dt_proba,
    'XGBoost_Stacked': stacked_pred
})
corr_models = model_predictions.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_models, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix of Model Predictions', fontsize=14)
plt.savefig('correlation_matrix_models.png')

# 📊 ROC Curves for All Models
plt.figure(figsize=(10, 8))
for name, proba in [('SVM', svm_proba), ('KNN', knn_proba), ('DecisionTree', dt_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# ROC for Stacked Model
stacked_proba = best_xgb.predict_proba(stacked_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, stacked_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'XGBoost Stacked (AUC = {roc_auc:.2f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')

# 📈 Precision-Recall Curves for All Models
plt.figure(figsize=(10, 8))
for name, proba in [('SVM', svm_proba), ('KNN', knn_proba), ('DecisionTree', dt_proba)]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.plot(recall, precision, label=f'{name}')

precision, recall, _ = precision_recall_curve(y_test, stacked_proba)
plt.plot(recall, precision, label=f'XGBoost Stacked', linestyle='--')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for All Models')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curves.png')

# Bar Plot of Model Performance Metrics
models = ['SVM', 'KNN', 'DecisionTree', 'XGBoost Stacked']
accuracies = [
    accuracy_score(y_test, svm_test),
    accuracy_score(y_test, knn_test),
    accuracy_score(y_test, dt_test),
    accuracy_score(y_test, stacked_pred)
]

plt.figure(figsize=(8, 6))
sns.barplot(x=accuracies, y=models, palette='viridis')
plt.xlabel('Accuracy')
plt.title('Model Performance Comparison (Accuracy)')
plt.xlim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center')
plt.savefig('model_performance_comparison.png')

# Performance Evaluation with Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')

print("\n===== Training Performance =====")
print("SVM Accuracy:", accuracy_score(y_train, best_svm.predict(X_train_scaled)))
print("k-NN Accuracy:", accuracy_score(y_train, best_knn.predict(X_train_scaled)))
print("Decision Tree Accuracy:", accuracy_score(y_train, best_dt.predict(X_train)))
print("XGBoost Stacking Accuracy:", accuracy_score(y_train, best_xgb.predict(stacked_train)))

print("\n===== Testing Performance =====")
print("SVM:\n", classification_report(y_test, svm_test))
print("k-NN:\n", classification_report(y_test, knn_test))
print("Decision Tree:\n", classification_report(y_test, dt_test))
print("XGBoost Stacking Model:\n", classification_report(y_test, stacked_pred))

print("SVM Accuracy:", accuracy_score(y_test, best_svm.predict(X_test_scaled)))
print("k-NN Accuracy:", accuracy_score(y_test, best_knn.predict(X_test_scaled)))
print("Decision Tree Accuracy:", accuracy_score(y_test, best_dt.predict(X_test)))
print("XGBoost Stacking Model Accuracy:", accuracy_score(y_test, stacked_pred))

# Plot confusion matrices for each model
plot_confusion_matrix(y_test, svm_test, "SVM Confusion Matrix")
plot_confusion_matrix(y_test, knn_test, "KNN Confusion Matrix")
plot_confusion_matrix(y_test, dt_test, "Decision Tree Confusion Matrix")
plot_confusion_matrix(y_test, stacked_pred, "XGBoost Stacked Confusion Matrix")

# Save model
model_data = {
    "model": best_xgb,
    "scaler": scaler,
    "encoders": label_encoders,
    "svm_model": best_svm,
    "knn_model": best_knn,
    "dt_model": best_dt
}

with open("asd_detection2.pkl", "wb") as file:
    pickle.dump(model_data, file)

# Load the saved model
with open("asd_detection2.pkl", "rb") as file:
    model_data = pickle.load(file)

print("Model, scaler, and encoders loaded successfully!")

# Extract components
model = model_data["model"]
scaler = model_data["scaler"]
label_encoders = model_data["encoders"]

# Example new input data
new_data = {
    "A1": 1, "A2": 0, "A3": 1, "A4": 0, "A5": 1,
    "A6": 0, "A7": 1, "A8": 0, "A9": 1, "A10": 0,
    "Age": 25, "Sex": "m", "Jauundice": "no", "Family_ASD": "no"
}

# Convert new input data to a DataFrame
new_df = pd.DataFrame([new_data])

# Encode categorical features
for col in ["Sex", "Jauundice", "Family_ASD"]:  
    new_df[col] = label_encoders[col].transform(new_df[col].str.lower())

# Scale the numerical feature (Age)
new_df["Age"] = scaler.transform(new_df[["Age"]])

# Generate base model predictions (SVM, k-NN, Decision Tree)
svm_pred = best_svm.predict_proba(new_df)[:, 1]
knn_pred = best_knn.predict_proba(new_df)[:, 1]
dt_pred = best_dt.predict_proba(new_df)[:, 1]

# Create stacked feature set for meta-model
stacked_input = pd.DataFrame({'SVM': svm_pred, 'KNN': knn_pred, 'DecisionTree': dt_pred})

# Make final prediction using XGBoost meta-model
prediction = best_xgb.predict(stacked_input)[0]

# Output result
result = "ASD" if prediction == 1 else "Non-ASD"
print("Prediction:", result)