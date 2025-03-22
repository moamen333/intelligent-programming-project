from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import dump
import pandas as pd
import numpy as np
data=pd.read_csv('heart.csv')
# Step 1: Split Data (80/20 train-test split)
X = data.drop(columns=['target'])  # Features (all columns except 'target')
y = data['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train a Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
# Step 3: Hyperparameter Tuning (Optimize tree depth and min samples per split)
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Use the best parameters found
best_dt = grid_search.best_estimator_
from sklearn.metrics import classification_report

# Load validation data (using the same test set for simplicity)
validation_data = X_test.copy()
validation_labels = y_test.copy()

# Evaluate the trained model
y_pred = best_dt.predict(validation_data)
print("Validation Metrics:")
print(classification_report(validation_labels, y_pred))
