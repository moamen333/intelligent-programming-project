# intelligent-programming-project
Heart Disease Risk Assessment System

Project Summary

This system is designed to evaluate the risk of heart disease in patients by combining machine learning techniques and expert system rules. It integrates a Decision Tree Classifier trained on medical data with a rule-based system derived from medical expertise.

Key Features

Utilizes a Decision Tree Classification model to predict heart disease risk.

Incorporates an expert system for rule-based evaluation of patient conditions.

Allows optimization of the machine learning model through hyperparameter tuning.

Provides comprehensive performance evaluation using classification metrics.

Offers visual representation of the Decision Tree to enhance interpretability.

Includes advanced data visualization for better insights.


Technologies and Libraries

The project is built with Python and uses the following tools:

pandas for data manipulation and preprocessing

numpy for numerical operations

scikit-learn to implement the Decision Tree model and hyperparameter tuning

experta to develop the rule-based expert system

matplotlib for plotting the Decision Tree structure

seaborn to create detailed and aesthetic visualizations

joblib for model persistence and reusability


Workflow

1. Data Handling:

Load and preprocess the dataset (heart.csv) for machine learning.

Split the data into training and testing subsets for model evaluation.



2. Model Implementation:

Train a Decision Tree Classifier using the training dataset.



3. Optimization:

Use Grid Search to fine-tune model parameters and improve accuracy.



4. Rule-Based Assessment:

Leverage medical knowledge to define rules for heart disease risk evaluation.



5. Analysis and Visualization:

Generate a classification report to measure model performance.

Visualize the Decision Tree structure and data trends using matplotlib and seaborn.
