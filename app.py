import streamlit as st
import joblib
import pandas as pd

# ------------------------------
# Load the trained model and scaler
# ------------------------------
model = joblib.load("titanic_survival_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction System")

st.write("""
Enter passenger details below to predict whether they survived the Titanic disaster.
""")

# ------------------------------
# User Inputs
# ------------------------------
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.2)

# ------------------------------
# Convert inputs into DataFrame
# ------------------------------
# Encode Sex (0 = Female, 1 = Male)
sex_encoded = 0 if sex == "Female" else 1

input_df = pd.DataFrame([[pclass, sex_encoded, age, sibsp, fare]],
                        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])

# Scale the input
input_scaled = scaler.transform(input_df)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.success(f"Prediction: {result}")
