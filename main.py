import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import time
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="AQI Bucket Prediction App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (same as before)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    </style>
""", unsafe_allow_html=True)

# load custom CSS for styling
def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css()

# Load the saved RandomForest model and scaler
try:
    model = joblib.load("aqi_random_forest_best_model.pkl")
    scaler = joblib.load("aqi_scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'aqi_random_forest_best_model.pkl' and 'aqi_scaler.pkl' are in the same directory.")
    st.stop()

# Load the original training data for visualizations (replace with your actual path)
try:
    original_data = pd.read_csv("city_day.csv")
    # Basic preprocessing for visualizations (handle NaNs as needed)
    original_data = original_data.fillna(original_data.median(numeric_only=True))
    if 'Date' in original_data.columns:
        original_data['Date'] = pd.to_datetime(original_data['Date'])
        original_data['Year'] = original_data['Date'].dt.year
        original_data['Month'] = original_data['Date'].dt.month
        original_data['DayOfWeek'] = original_data['Date'].dt.dayofweek
except FileNotFoundError:
    st.warning("Original data file 'city_day.csv' not found. Visualizations will be limited.")
    original_data = None

# --- Sidebar for Menu ---
menu = ["Predict", "Insights"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Predict":
    st.title("AQI Bucket Prediction App 🌍")
    st.write("Welcome! Enter the air pollutant concentrations and date below to predict the Air Quality Index (AQI) bucket category. Understand your air quality better!")

    # Using columns for better layout of inputs
    st.header("Enter Pollutant Concentrations")
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, step=1.0, help="Particulate Matter 2.5")
        no = st.number_input("NO (μg/m³)", min_value=0.0, step=1.0, help="Nitric Oxide")
        nh3 = st.number_input("NH3 (μg/m³)", min_value=0.0, step=1.0, help="Ammonia")
        o3 = st.number_input("O3 (μg/m³)", min_value=0.0, step=1.0, help="Ozone")

    with col2:
        pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, step=1.0, help="Particulate Matter 10")
        no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, step=1.0, help="Nitrogen Dioxide")
        co = st.number_input("CO (mg/m³)", min_value=0.0, step=0.01, help="Carbon Monoxide")
        benzene = st.number_input("Benzene (μg/m³)", min_value=0.0, step=0.01, help="Benzene")

    with col3:
        nox = st.number_input("NOx (μg/m³)", min_value=0.0, step=1.0, help="Nitrogen Oxides")
        so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, step=1.0, help="Sulfur Dioxide")
        toluene = st.number_input("Toluene (μg/m³)", min_value=0.0, step=0.01, help="Toluene")
        xylene = st.number_input("Xylene (μg/m³)", min_value=0.0, step=0.01, help="Xylene")

    st.header("Select Date")
    date_input = st.date_input(
        "Date of Measurement",
        min_value=datetime.date(2000, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )
    year = date_input.year
    month = date_input.month
    day = date_input.day
    dayofweek = date_input.weekday()

    st.markdown("---") # Separator

    if st.button("Predict AQI Bucket"):
        with st.spinner("Predicting..."):
            time.sleep(2)
        # Create DataFrame with numerical inputs
        numeric_data = np.array([[pm25, pm10, no, no2, nox, nh3,
                                    co, so2, o3, benzene, toluene, xylene]])

        # Scale numerical data
        numeric_scaled = scaler.transform(numeric_data)

        # Combine with date features
        date_features = np.array([[year, month, day, dayofweek]])
        final_input = np.concatenate([numeric_scaled, date_features], axis=1)

        # Predict
        prediction = model.predict(final_input)[0]

        # Color-coded result based on severity with enhanced styling
        prediction_lower = prediction.lower()
        if prediction_lower == "good":
            st.markdown(
                f"<div class='prediction-box good'>"
                f"<h3>Predicted AQI Bucket: {prediction} ✨</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif prediction_lower == "satisfactory":
            st.markdown(
                f"<div class='prediction-box satisfactory'>"
                f"<h3>Predicted AQI Bucket: {prediction} 👍</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif prediction_lower == "moderate":
            st.markdown(
                f"<div class='prediction-box moderate'>"
                f"<h3>Predicted AQI Bucket: {prediction} ⚠️</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif prediction_lower == "poor":
            st.markdown(
                f"<div class='prediction-box poor'>"
                f"<h3>Predicted AQI Bucket: {prediction} 😷</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif prediction_lower == "very poor":
            st.markdown(
                f"<div class='prediction-box very-poor'>"
                f"<h3>Predicted AQI Bucket: {prediction} 🚨</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif prediction_lower == "severe":
            st.markdown(
                f"<div class='prediction-box severe'>"
                f"<h3>Predicted AQI Bucket: {prediction} 🛑</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='prediction-box default'>"
                f"<h3>Predicted AQI Bucket: {prediction}</h3>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.info("This app uses a pre-trained RandomForest model to predict AQI buckets based on pollutant concentrations and date features.")

elif choice == "Insights":
    st.title("AQI Data Insights 📊")
    st.write("Explore trends and distributions of air pollutants in the dataset.")

    if original_data is not None and 'Date' in original_data.columns:
        st.subheader("Pollutant Distributions")
        pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3','Benzene', 'Toluene', 'Xylene']
        selected_pollutant = st.selectbox("Select a Pollutant for Distribution Analysis", pollutants)
        fig_dist = px.histogram(original_data, x=selected_pollutant, nbins=30, title=f'Distribution of {selected_pollutant}')
        st.plotly_chart(fig_dist)

        st.subheader(f"Trend of {selected_pollutant} Over Time")

        # Original pollutant level line
        fig_trend = px.line(original_data, x='Date', y=selected_pollutant,
                             title=f'Trend of {selected_pollutant} Over Time',
                             labels={'Date': 'Date', selected_pollutant: f'{selected_pollutant} (μg/m³ or mg/m³)'},
                             color_discrete_sequence=['coral']) # Keep the original line color

        st.plotly_chart(fig_trend)

        st.subheader("AQI Bucket Distribution Over Time (Yearly)")
        if 'Year' in original_data.columns and 'AQI_Bucket' in original_data.columns:
            yearly_aqi = original_data.groupby('Year')['AQI_Bucket'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
            fig_yearly_aqi = px.bar(yearly_aqi, x='Year', y='Percentage', color='AQI_Bucket',
                                     title='Yearly Distribution of AQI Buckets', labels={'Percentage': 'Percentage (%)'})
            st.plotly_chart(fig_yearly_aqi)
        else:
            st.warning("Required columns ('Year', 'AQI_Bucket') not found for yearly AQI analysis.")

        st.subheader("Average Pollutant Levels by Day of the Week")
        if 'DayOfWeek' in original_data.columns:
            day_mapping = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            original_data['DayOfWeekName'] = original_data['DayOfWeek'].map(day_mapping)
            avg_pollutant_day = original_data.groupby('DayOfWeekName')[pollutants].mean().reset_index()
            melted_day = pd.melt(avg_pollutant_day, id_vars=['DayOfWeekName'], var_name='Pollutant', value_name='Average Level')
            fig_avg_day = px.line(melted_day, x='DayOfWeekName', y='Average Level', color='Pollutant',
                                   title='Average Pollutant Levels by Day of the Week',
                                   labels={'DayOfWeekName': 'Day of the Week', 'Average Level': 'Average Level (μg/m³ or mg/m³)'})
            st.plotly_chart(fig_avg_day)
        else:
            st.warning("Required column ('DayOfWeek') not found for day-wise pollutant analysis.")

    else:
        st.info("Please ensure the 'city_day.csv' file is available to display insights.")