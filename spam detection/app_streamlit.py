# app_streamlit.py
import streamlit as st
import joblib

# Load pipeline
pipeline = joblib.load("pipeline.pkl")

st.title("SMS Spam Detector")

text = st.text_area("Enter your message:")

if st.button("Predict"):
    prediction = pipeline.predict([text])[0]
    st.write("Prediction:", prediction)