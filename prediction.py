import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    return df
def predict_future_data(df, forecast_period):
    df = df.set_index('timestamp')
    
    # extract hourly patterns
    df['hour'] = df.index.hour
    hourly_pattern = df.groupby('hour').mean()
    # create future timestamps
    future_dates = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=forecast_period, freq='H')
    # list to hold predictions 
    future_num_people = []
    future_temperatures = []
    # Prediction
    for date in future_dates:
        hour = date.hour
        num_people = hourly_pattern.loc[hour, 'num_people']
        temperature = hourly_pattern.loc[hour, 'temperature']
        future_num_people.append(num_people)
        future_temperatures.append(temperature)
    future_df = pd.DataFrame({'timestamp': future_dates, 'num_people': future_num_people, 'temperature': future_temperatures})
    return future_df
st.title("Future Data Predictions based on given data ")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)

    st.write("### Loaded Data")
    st.write(data.head())
    forecast_period = st.slider("Select forecast period (hours)", 1, 24)

    if st.button("Predict Future Data"):
        future_data = predict_future_data(data, forecast_period)
        st.write("### Future Predictions")
        st.write(future_data)
        
        # Plotting the results
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        plt.plot(data['timestamp'], data['temperature'], label='Actual Temperature', color='blue', linestyle='-', marker='o')
        plt.plot(future_data['timestamp'], future_data['temperature'], label='Predicted Temperature', color='red', linestyle='--', marker='x')
        plt.plot(data['timestamp'], data['num_people'], label='Actual Num People', color='green', linestyle='-', marker='o')
        plt.plot(future_data['timestamp'], future_data['num_people'], label='Predicted Num People', color='orange', linestyle='--', marker='x')
        plt.xlabel("Timestamp")
        plt.ylabel("Values")
        plt.title("Temperature and Number of People vs. Timestamp")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
