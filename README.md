# ANN-Clasification-Customer-Churn-Prediction
Customer Churn Prediction App

This project is a Streamlit-based web application that predicts the likelihood of a customer churning (leaving the bank), using a trained Artificial Neural Network (ANN) model. It is designed to provide an intuitive UI for bank staff to input customer data and receive real-time churn predictions.

Features

- Predicts churn probability using a trained ANN model (`model.h5`)
- Encodes categorical variables using pre-trained encoders (`LabelEncoder`, `OneHotEncoder`)
- Scales input features using a pre-fitted `StandardScaler`
- Interactive UI built with Streamlit app takes user input to display results
- Real-time results display
