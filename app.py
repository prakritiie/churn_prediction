
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

#load the ohe
with open('onehot_encoder_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)


#load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#streamlit app

st.title('Customer Churn Prediction')
st.write('Enter customer details to predict if they are likely to churn.')

# Input fields for customer details
gender = st.selectbox('Gender', label_encoder_gender.classes_)
geography = st.selectbox('Geography', list(ohe_geo.categories_[0]))
age = st.slider('Age', min_value=18, max_value=95)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.number_input('Tenure', min_value=0, max_value=10)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4)
has_cr_card = st.selectbox('Has Credit Card', options=[0, 1])
is_active_member = st.selectbox('Is Active Member', options=[0, 1])


#prepare the input data
input_data = ({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
   'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],

    'Geography': [geography]

})

# geography
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

#combine

input_data = pd.concat([pd.DataFrame(input_data), geo_encoded], axis=1)
input_data = input_data.drop('Geography', axis=1)
# Ensure column order matches scaler expectation
input_data = input_data[scaler.feature_names_in_]
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]


#prediction button
if st.button('Predict Churn'):
    st.write('Predicting...')
    if prediction_proba > 0.5:
        st.write(f'The customer is likely to churn with a probability of {prediction_proba:}')
    else:
        st.write(f'The customer is unlikely to churn with a probability of {prediction_proba:}')

