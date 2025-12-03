import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

model = pickle.load(open('Linear_Regression_Model.pkl', 'rb'))
st.title('Linear Regression Model')
tv = st.text_input('Enter TV')
radio = st.text_input('Enter Radio')
newspaper = st.text_input('Enter Newspaper')

if st.button('Predict'):
    features = np.array([[tv,radio, newspaper]], dtype=np.float64)
    prediction = model.predict(features).reshape(1,-1)
    st.write("prediction Value is", prediction[0])