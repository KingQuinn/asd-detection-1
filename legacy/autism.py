####################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split


# Load the dataset
df = pd.read_csv("Autism_Dataset_Combined.csv")  # Replace with your actual dataset

#  Check for duplicates
df.drop_duplicates(inplace=True)

# Handle categorical data
categorical_cols = df.select_dtypes(include=['object']).columns  # Identify categorical columns

# Apply Label Encoding for binary categorical features
label_encoders = {}
for col in categorical_cols:
    if df[col].nunique() == 2:  # Binary categorical features
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Apply One-Hot Encoding for nominal categorical features (more than two unique values)
df = pd.get_dummies(df, columns=[col for col in categorical_cols if df[col].nunique() > 2], drop_first=True)

# Handle numerical data
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Remove Outliers using Z-score method
z_scores = np.abs((df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std())
df = df[(z_scores < 3).all(axis=1)]  

#correlation matrix
#RANDOM FOREST feature importance scores

# Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("Data cleaning complete. Cleaned dataset saved.")

df = pd.read_csv("cleaned_dataset.csv")

# Split features and target
X = df.drop(columns=["Class"]) # features
y = df["Class"] # target


# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display split sizes
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")


# Define hyperparameter grids
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
param_grid_dt = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}

# Perform grid search for each model
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=5, scoring='accuracy')
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')

# Train grid search models
svm_grid.fit(X_train, y_train)
knn_grid.fit(X_train, y_train)
dt_grid.fit(X_train, y_train)

# Get best estimators
best_svm = svm_grid.best_estimator_
best_knn = knn_grid.best_estimator_
best_dt = dt_grid.best_estimator_

# Train optimized base models
best_svm.fit(X_train, y_train)
best_knn.fit(X_train, y_train)
best_dt.fit(X_train, y_train)

# Generate cross-validated predictions for stacking
svm_pred = cross_val_predict(best_svm, X_train, y_train, cv=5, method="predict_proba")[:, 1]
knn_pred = cross_val_predict(best_knn, X_train, y_train, cv=5, method="predict_proba")[:, 1]
dt_pred = cross_val_predict(best_dt, X_train, y_train, cv=5, method="predict_proba")[:, 1]

# Stack predictions for meta-model
stacked_train = pd.DataFrame({'SVM': svm_pred, 'KNN': knn_pred, 'DecisionTree': dt_pred})

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_train)

# Generate test set predictions
svm_test_proba = best_svm.predict_proba(X_test)[:, 1]
knn_test_proba = best_knn.predict_proba(X_test)[:, 1]
dt_test_proba = best_dt.predict_proba(X_test)[:, 1]

# Stack test set predictions
stacked_test = pd.DataFrame({'SVM': svm_test_proba, 'KNN': knn_test_proba, 'DecisionTree': dt_test_proba})

# Make final predictions with meta-model
stacked_predictions = meta_model.predict(stacked_test)

# Train XGBoost as the meta-model
meta_model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
meta_model_xgb.fit(stacked_train, y_train)

# Make predictions using XGBoost meta-model
stacked_predictions_xgb = meta_model_xgb.predict(stacked_test)


# Evaluate models
# Accuracy comparison
print("SVM Accuracy:", accuracy_score(y_test, best_svm.predict(X_test)))
print("k-NN Accuracy:", accuracy_score(y_test, best_knn.predict(X_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, best_dt.predict(X_test)))
print("Stacking Model Accuracy:", accuracy_score(y_test, stacked_predictions)) 

# Evaluate performance of stacking model and XGBoost meta-model
print("XGBoost Meta-Model Performance:\n", classification_report(y_test, stacked_predictions_xgb))
print("XGBoost Stacking Model Accuracy:", accuracy_score(y_test, stacked_predictions_xgb))
