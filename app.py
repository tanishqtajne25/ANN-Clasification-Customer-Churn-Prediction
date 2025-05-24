import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("This app predicts whether a bank customer is likely to churn based on their profile. "
                "Enter the details on the main page and click **Predict**.")

# App title
st.title('💼 Customer Churn Prediction')

# Form for user input
with st.form("churn_form"):
    st.header("🔍 Enter Customer Details")

    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    balance = st.number_input('Balance', min_value=0.0)
    credit_score = st.number_input('Credit Score', min_value=0.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
    tenure = st.slider('Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine all features
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        prediction_proba = prediction[0][0]

        # Display results
        st.markdown(f"### 🔢 Churn Probability: `{prediction_proba:.2%}`")

        if prediction_proba > 0.5:
            st.markdown("### ❌ The customer is **likely to churn**.")
        else:
            st.markdown("### ✅ The customer is **not likely to churn**.")
