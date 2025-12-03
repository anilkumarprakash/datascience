import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
sc = StandardScaler()
le = LabelEncoder()
# Load model and data
model = pickle.load(open('_06_logist_regression.pkl', 'rb'))



# create web app

st.title("Logistic Regression for Churn Prediction")
gender = st.selectbox("Select Gender",  options=['Male', 'Female'])
SeniorCitizen = st.selectbox("Are you Senior Citizen?",   options=['No', 'Yes'])
Partner = st.selectbox("Do you have Partner?",  options=['Yes', 'No'])
Dependents = st.selectbox("Are you Dependents on others?",  options=['No', 'Yes'])
tenure = st.text_input("What is your tenure?")
PhoneService = st.selectbox("Do you have Phone Service", options=['Yes', 'No'])
multiline = st.selectbox("Do you have Multilingual Service?", options=['Yes', 'No', 'No phone service'])
Contract = st.selectbox("Your contracts?", options=['Month-to-month', 'One Year' , 'Two Year'])
TotalCharges = st.text_input("Enter your Total Charges")


# helper function
def prediction(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges):
    data = {
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Dependents': [Dependents],
    'Partner': [Partner],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'Contract': [Contract],
    'TotalCharges': [TotalCharges]
    }
    # Create a DataFrame from the dictionary
    dff = pd.DataFrame(data)
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','tenure', 'PhoneService', 'MultipleLines', 'Contract', 'TotalCharges']
    for column in categorical_columns:
        dff[column] = le.fit_transform(dff[column])
    dff = sc.fit_transform(dff)

    result = model.predict(dff).reshape(1,-1)
    return result[0]

# button

if st.button("Predict"):
    gender = "Female"
    Seniorcitizen = "No"
    Partner = "Yes"
    Dependents = "No"
    tenure = 1
    Phoneservice = "No"
    multiline = "No phone service"
    contact = "Month-to-month"
    totalcharge = 29.85

    result = prediction(gender, Seniorcitizen, Partner, Dependents, tenure, Phoneservice, multiline, contact,
                        totalcharge)

    if result == 0:
        st.write("Not Churn")
    else:
        st.write("Churn")




