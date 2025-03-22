from experta import KnowledgeEngine, Rule, Fact, P

# Define an Expert System Class
class HeartDisease(KnowledgeEngine):
    @Rule(Fact(exercise=P(lambda x: x == "regular")) & Fact(BMI=P(lambda x: x < 25)))
    def low_risk(self):
        print("Low Risk: Regular exercise and healthy BMI indicate low risk.")

    @Rule(Fact(bloodPressure=P(lambda x: x > 140)) & Fact(smoking=P(lambda x: x == "yes")))
    def high_risk_smoking(self):
        print("Warning: High Risk due to high blood pressure and smoking.")

    @Rule(Fact(cholesterol=P(lambda x: x > 240)) & Fact(age=P(lambda x: x > 50)))
    def high_cholesterol_risk(self):
        print("High Risk: High cholesterol and age indicate a risk of heart disease.")

    @Rule(Fact(bloodPressure=P(lambda x: x > 160)))
    def very_high_bp(self):
        print("Critical Risk: Extremely high blood pressure requires immediate attention.")

    @Rule(Fact(BMI=P(lambda x: x > 30)) & Fact(exercise=P(lambda x: x == "none")))
    def obesity_risk(self):
        print("Warning: Obesity and lack of exercise increase heart disease risk.")

    @Rule(Fact(gender="male") & Fact(age=P(lambda x: x > 45)))
    def male_risk(self):
        print("Moderate Risk: Male above 45 years old has an elevated risk.")

    @Rule(Fact(gender="female") & Fact(age=P(lambda x: x > 55)))
    def female_risk(self):
        print("Moderate Risk: Female above 55 years old has an elevated risk.")

    @Rule(Fact(cholesterol=P(lambda x: x <= 200)) & Fact(BMI=P(lambda x: x < 25)))
    def healthy_profile(self):
        print("Low Risk: Healthy cholesterol level and BMI.")

    @Rule(Fact(family_history="yes") & Fact(smoking=P(lambda x: x == "yes")))
    def family_history_risk(self):
        print("High Risk: Family history combined with smoking increases risk.")

    @Rule(Fact(diabetes="yes") & Fact(cholesterol=P(lambda x: x > 220)))
    def diabetes_cholesterol_risk(self):
        print("High Risk: Diabetes and high cholesterol increase the likelihood of heart disease.")

    # Default case for no detected risks
    @Rule()
    def default(self):
        print("No significant risk factors detected.")

# Create an instance of the expert system
engine = HeartDisease()

# Reset the engine to clear any previous state
engine.reset()