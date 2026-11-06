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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
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

# 🔍 Correlation Matrix Plot
corr = df.corr()

# Prepare data for 3D plot
features = corr.columns
x, y = np.meshgrid(np.arange(len(features)), np.arange(len(features)))
z = corr.values

# Create 3D figure
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection='3d')

# Create surface plot
surf = ax.plot_surface(x, y, z, cmap='coolwarm',
                      rstride=1, cstride=1,
                      edgecolor='none',
                      alpha=0.8)

# Customize axes
ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(features)))
ax.set_xticklabels(features, rotation=90, fontsize=8)
ax.set_yticklabels(features, fontsize=8)
ax.set_zlabel('Correlation Strength', labelpad=10)

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

# Add title and adjust view
ax.set_title('3D Correlation Matrix of ASD Features', pad=20, fontsize=14)
ax.view_init(elev=30, azim=45)  # Adjust viewing angle

# Add grid lines for better readability
ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)
ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.2)

plt.tight_layout()
plt.show()

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
plt.show()


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

# ✅ Performance Evaluation
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

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



