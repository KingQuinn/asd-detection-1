import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Autism_Dataset_Combined.csv")  

#  Check for duplicates
df.drop_duplicates(inplace=True)

# Handle categorical data
categorical_cols = df.select_dtypes(include=['object']).columns  # Identify categorical columns

# Apply Label Encoding for binary categorical features
label_encoders = {}
for col in categorical_cols:
    if df[col].nunique() == 2:  # Binary categorical features
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].str.lower())
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

# Load the cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Separate features and target variable
X = df.drop(columns=["Class"])
y = df["Class"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply feature scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Identify numerical features (assuming 'Age' is numerical)
numeric_features = ['Age']
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Define hyperparameter grids
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
param_grid_dt = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

# Perform grid search for each model
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=5, scoring='accuracy')
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
xgb_grid = GridSearchCV(XGBClassifier(random_state=42), param_grid_xgb, cv=5, scoring='accuracy')

# Train grid search models
svm_grid.fit(X_train_scaled, y_train)
knn_grid.fit(X_train_scaled, y_train)
dt_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)

# Get best estimators
best_svm = svm_grid.best_estimator_
best_knn = knn_grid.best_estimator_
best_dt = dt_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

# Generate cross-validated predictions for stacking
svm_pred = cross_val_predict(best_svm, X_train_scaled, y_train, cv=5, method="predict_proba")[:, 1]
knn_pred = cross_val_predict(best_knn, X_train_scaled, y_train, cv=5, method="predict_proba")[:, 1]
dt_pred = cross_val_predict(best_dt, X_train, y_train, cv=5, method="predict_proba")[:, 1]

# Stack predictions for meta-model
stacked_train = pd.DataFrame({'SVM': svm_pred, 'KNN': knn_pred, 'DecisionTree': dt_pred})

# Train meta-model (XGBoost)
best_xgb.fit(stacked_train, y_train)

# Generate test set predictions
svm_test_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
knn_test_proba = best_knn.predict_proba(X_test_scaled)[:, 1]
dt_test_proba = best_dt.predict_proba(X_test)[:, 1]

# Stack test set predictions
stacked_test = pd.DataFrame({'SVM': svm_test_proba, 'KNN': knn_test_proba, 'DecisionTree': dt_test_proba})

# Make final predictions with XGBoost meta-model
stacking_model= best_xgb.predict(stacked_test)

# Evaluate models
print("SVM Performance:\n", classification_report(y_test, best_svm.predict(X_test_scaled)))
print("k-NN Performance:\n", classification_report(y_test, best_knn.predict(X_test_scaled)))
print("Decision Tree Performance:\n", classification_report(y_test, best_dt.predict(X_test)))
print("XGBoost Stacking Model Performance:\n", classification_report(y_test, stacking_model))

# Accuracy comparison
print("SVM Accuracy:", accuracy_score(y_test, best_svm.predict(X_test_scaled)))
print("k-NN Accuracy:", accuracy_score(y_test, best_knn.predict(X_test_scaled)))
print("Decision Tree Accuracy:", accuracy_score(y_test, best_dt.predict(X_test)))
print("XGBoost Stacking Model Accuracy:", accuracy_score(y_test, stacking_model))


# Save the XGBoost meta-model
import pickle
# Create a dictionary containing the model, scaler, and encoders
model_data = {
    "model": best_xgb,  
    "scaler": scaler,  
    "encoders": label_encoders, # Dictionary of encoders for categorical features
    "svm_model": best_svm,
    "knn_model": best_knn,
    "dt_model": best_dt
}

# Save everything into a single pickle file
with open("asd_detection.pkl", "wb") as file:
    pickle.dump(model_data, file)

# Load the saved model
with open("asd_detection.pkl", "rb") as file:
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
