import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model

def create_model(optimizer='adam'):
    visible = Input(shape=(len(important_features),))
    hidden1 = Dense(10, activation='relu')(visible)
    hidden2 = Dense(20, activation='relu')(hidden1)
    hidden3 = Dense(10, activation='relu')(hidden2)
    output = Dense(1, activation='sigmoid')(hidden3)
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define your list of important features
important_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']

model = joblib.load('best_keras_model.pkl')  # Load your Keras model
scaler = joblib.load('standard_scaler.pkl')  # Load your scaler

# Streamlit app code
def main():
    st.title('Customer Churn Prediction')

    st.write('Enter customer details to predict churn')

    # Input fields for user to enter customer details
    tenure = st.number_input('Tenure')
    monthly_charges = st.number_input('Monthly Charges')
    total_charges = st.number_input('Total Charges')
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Map contract and payment method to numerical values
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_method_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}

    # Preprocess input data
    features = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract_map[contract]],
        'PaymentMethod': [payment_method_map[payment_method]]
    })

    # Scale the input features
    scaled_features = scaler.transform(features)

    if st.button('Predict'):
        prediction = model.predict(np.array(scaled_features))
        confidence_score = prediction[0]
        churn_prediction = 1 if confidence_score > 0.5 else 0
        st.success(f'The predicted probability of churn is: {confidence_score:.4f}')

        if churn_prediction == 1:
            st.write(f'Churn Prediction: Likely to churn with confidence score: {confidence_score:.2f}')
        else:
            st.write(f'Churn Prediction: Not likely to churn with confidence score: {confidence_score:.2f}')

if __name__ == '__main__':
    main()
