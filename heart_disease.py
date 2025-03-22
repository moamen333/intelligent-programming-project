import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data=pd.read_csv('heart.csv')
print(data.isnull().sum())
data.fillna(data.mean(),inplace=True)
print(data)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
print("Normalized Data:")
print(normalized_data)



#dataset processing1
#data normalization 
#data visualization 2
#implementation for expert systems

from experta import KnowledgeEngine, Rule, Fact, P # Import Rule, Fact, and P

# Define an Expert System Class
class HeartDisease(KnowledgeEngine):

    @Rule(Fact(exercise=P(lambda x: x =="regular"))& Fact(BMI=P(lambda x:x <25)))
    def high_glucose(self):
        print("Low risk")

   

    @Rule(Fact(bloodPressure=P(lambda x: x > 140))&Fact(smoking=P(lambda x:x =="yes")))
    def genetic_risk(self):
        print("Warning:High Risk")

    @Rule(Fact(chlosterol=P(lambda x: x > 240)) & Fact(age=P(lambda x: x > 50)) )
    def high_diabetes_risk(self):
        print("High risk of Heart Disease! Consult a doctor.")

    @Rule(Fact(target=1))
    def diabetic(self):
        print("This patient has diabetes.")


engine = HeartDisease()
engine.reset()
patient = {
    "chol": 4,
    "exercise": "regular",
    "fbs": 0,
    "smoking": "yes",
    "bloodPressure": 210,
    "bmi": 32,
    
    "age": 60,
    "target": 1  # Known diagnosis from dataset
}

for key, value in patient.items():
    engine.declare(Fact(**{key: value}))

engine.run()


X = data.drop(columns=['target'])  # Features
y = data['target']  # Target (Diabetic or Not)
print(X)
print(y)