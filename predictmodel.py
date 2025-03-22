from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import dump
from trainmodel import validation_data,validation_labels,best_dt,X_test,X_train,y_test,y_train
# Step 2: Predictions from Decision Tree
dt_predictions = best_dt.predict(validation_data)

# Step 3: Predictions from Expert System
def expert_system_predict(row):
    # Rule 1: High risk if oldpeak > 2.0 and ca > 1
    if row['oldpeak'] > 2.0 and row['ca'] > 1:
        return 1  # High risk
    
    # Rule 2: Low risk if thalach > 150 and no exercise-induced angina
    elif row['thalach'] > 150 and row['exang'] == 0:
        return 0  # Low risk
    
    # Rule 3: Default to moderate risk
    else:
        return 0  # Moderate risk


# Apply the expert system to validation data
expert_predictions = validation_data.apply(expert_system_predict, axis=1)

# Step 4: Compare Metrics
print("Decision Tree Model Performance:")
print(classification_report(validation_labels, dt_predictions))

print("Expert System Performance:")
print(classification_report(validation_labels, expert_predictions))





# Step 4: Evaluate the Model
# Make predictions
y_pred = best_dt.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save the trained model using joblib
dump(best_dt, 'decision_tree_model.joblib')

# Step 5: Discuss Explainability
# Decision Tree explainability
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(best_dt, feature_names=validation_data.columns, class_names=["No Risk", "High Risk"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Print insights about the comparison
print("Explainability Notes:")
print("""
- Decision Tree:
    - Derived from training data, adapts to patterns in the dataset.
    - Can be visualized and analyzed to understand decision-making.
- Expert System:
    - Based on human-defined rules, limited by the scope of domain knowledge.
    - More interpretable but less flexible than data-driven models.
""")
