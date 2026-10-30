import streamlit as st
import joblib

app = joblib.load("models/diebetes.pkl")

st.title("DIEBETES PRIDICTION")

Glucose = st.number_input("Glucose Level")
BMI = st.number_input("BMI Level")
Age = st.number_input("Age")

#"Glucose","BMI","Age"
if st.button("Predict"):
    result = app.predict([[Glucose,BMI,Age]])
    #st.write(result)
    result_text = "Yes" if result == 1 else "No"
    st.header(f"Prediction : {result_text}")
  
