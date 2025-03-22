from experta import KnowledgeEngine, Rule, Fact, P
from rules import HeartDisease
# Define user input for risk assessment
def get_user_input():
    # Ask user for their input
    age = int(input("Enter your age: "))
    blood_pressure = int(input("Enter your blood pressure (e.g., 120): "))
    cholesterol = int(input("Enter your cholesterol level (e.g., 200): "))
    BMI = float(input("Enter your BMI (e.g., 24.5): "))
    exercise = input("Do you exercise regularly (yes/no): ").lower()
    smoking = input("Do you smoke (yes/no): ").lower()
    family_history = input("Do you have a family history of heart disease (yes/no): ").lower()
    diabetes = input("Do you have diabetes (yes/no): ").lower()

    # Return a dictionary with user input
    return {
        "age": age,
        "bloodPressure": blood_pressure,
        "cholesterol": cholesterol,
        "BMI": BMI,
        "exercise": "regular" if exercise == "yes" else "none",
        "smoking": smoking,
        "family_history": family_history,
        "diabetes": diabetes,
    }

def predict_risk(patient_data):
    print(type(patient_data), patient_data)  # Debugging
    if patient_data["cholesterol"] > 240 and patient_data["bloodPressure"] > 140:
        print("This patient is at HIGH risk of heart disease.")
    elif patient_data["BMI"] < 25 and patient_data["exercise"] == "regular":
        print("This patient is at LOW risk of heart disease.")
    else:
        print("This patient is at MODERATE risk of heart disease.")


# Gather user input
patient_data = get_user_input()

# Predict risk based on user input
predict_risk(patient_data)

# Create an instance of the expert system
engine = HeartDisease()
engine.reset()

# Declare facts based on patient data
for key, value in patient_data.items():
    engine.declare(Fact(**{key: value}))


# Run the inference engine
engine.run()

# Run the inference engine




