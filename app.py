import pandas as pd
import joblib
import streamlit as st
import numpy as np
import pickle

model = joblib.load('portfolio_prediction_model.pkl')

# Title and Subheader
st.title('📊 Portfolio Return Prediction App')
st.subheader("Predict portfolio returns based on key financial indicators")

# Sidebar for input
st.sidebar.header("Input the following details")

# Create sliders for each feature matching your dataset
asset_allocation = st.sidebar.slider('Asset Allocation (%)', min_value=40.0, max_value=100.0, step=0.1, value=60.0)
historical_volatility = st.sidebar.slider('Historical Volatility (%)', min_value=5.0, max_value=25.0, step=0.1, value=10.0)
gdp_growth_rate = st.sidebar.slider('GDP Growth Rate (%)', min_value=1.0, max_value=6.0, step=0.1, value=3.0)
inflation_rate = st.sidebar.slider('Inflation Rate (%)', min_value=1.0, max_value=5.0, step=0.1, value=2.5)
interest_rate = st.sidebar.slider('Interest Rate (%)', min_value=0.5, max_value=4.0, step=0.1, value=2.0)
market_return = st.sidebar.slider('Market Return (%)', min_value=-10.0, max_value=20.0, step=0.1, value=5.0)

# Predict button in a prominent color
if st.button('🚀 Predict Portfolio Return'):
    # Prepare the input data as per the column names
    input_data = np.array([[asset_allocation, historical_volatility, gdp_growth_rate,
                            inflation_rate, interest_rate, market_return]])

    # Make a prediction using the loaded model
    predicted_return = model.predict(input_data)[0]

    # Display the predicted portfolio return
    st.write(f'### Predicted Portfolio Return: **{predicted_return:.2f}%**')

# Footer
st.markdown("""
    ---
    <div style='text-align:center'>
    <small>Powered by Streamlit • Developed with 💻 and ❤️</small>
    </div>
    """, unsafe_allow_html=True)
