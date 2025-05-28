import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="AQI Bucket Prediction App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0; /* Default light grey text color for general content on dark background */
    }

    /* Apply a dark background color to the main app container */
    div[data-testid="stAppViewContainer"] {
        background-color: #222831 !important; /* Deep charcoal background */
        background-image: none !important; /* Ensure no background image */
    }

    /* Target the main content block to give it a distinct card-like appearance */
    .main {
        background-color: #393e46 !important; /* Slightly lighter dark grey for the main content area */
        padding: 30px; /* Increased padding */
        border-radius: 15px; /* More rounded corners */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25); /* Stronger, softer shadow for dark theme */
        margin-top: 20px; /* Add some space from the top */
        margin-bottom: 20px; /* Add some space at the bottom */
    }

    /* Target for the main title container (Streamlit's default header container) */
    .st-emotion-cache-1cypcdb {
        background-color: #222831; /* Match overall dark background */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 30px;
        text-align: center;
    }

    /* Ensure h1 and h3 have very light text for maximum contrast on dark background */
    h1 {
        color: #ffffff !important; /* White for main title */
        font-weight: 700 !important;
        font-size: 2.5em !important;
        margin-bottom: 10px !important;
    }

    h3 {
        color: #ffffff !important; /* White for subheadings */
        font-weight: 600 !important;
        font-size: 1.5em !important;
    }

    /* Target for st.write text (main description) */
    .st-emotion-cache-nahz7x p,
    .st-emotion-cache-nahz7x .stText {
        color: #cccccc !important; /* Light grey for better readability on dark background */
        font-size: 1.1em !important;
        line-height: 1.6 !important;
    }

    /* Styling for Streamlit input elements - targeting the container divs for overall box styling */
    div[data-testid*="stNumberInput"] > div:first-child > div:first-child,
    div[data-testid*="stDateInput"] > div:first-child > div:first-child,
    div[data-testid*="stTextInput"] > div:first-child > div:first-child,
    div[data-testid*="stSelectbox"] > div:first-child > div:first-child,
    div[data-testid*="stSlider"] > div:first-child > div:first-child {
        background-color: #444b54 !important; /* Darker grey for input boxes */
        border-radius: 8px !important;
        border: 1px solid #5a626a !important; /* Slightly darker, subtle border */
        padding: 8px 12px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15) !important; /* Subtle shadow */
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    /* Hover and Focus for the input containers */
    div[data-testid*="stNumberInput"] > div:first-child > div:first-child:hover,
    div[data-testid*="stDateInput"] > div:first-child > div:first-child:hover,
    div[data-testid*="stTextInput"] > div:first-child > div:first-child:hover,
    div[data-testid*="stSelectbox"] > div:first-child > div:first-child:hover,
    div[data-testid*="stSlider"] > div:first-child > div:first-child:hover {
        border-color: #6a82a0 !important; /* Lighter blue-grey on hover */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important; /* Enhanced shadow on hover */
    }

    div[data-testid*="stNumberInput"] > div:first-child > div:first-child:focus-within,
    div[data-testid*="stDateInput"] > div:first-child > div:first-child:focus-within,
    div[data-testid*="stTextInput"] > div:first-child > div:first-child:focus-within,
    div[data-testid*="stSelectbox"] > div:first-child > div:first-child:focus-within,
    div[data-testid*="stSlider"] > div:first-child > div:first-child:focus-within {
        border-color: #3498db !important; /* Blue on focus */
        box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.4) !important; /* Stronger blue glow on focus */
        outline: none !important;
    }

    /* Also target the actual input/textarea elements inside these containers */
    div[data-testid*="stNumberInput"] input,
    div[data-testid*="stDateInput"] input,
    div[data-testid*="stTextInput"] input,
    div[data-testid*="stTextInput"] textarea {
        background-color: transparent !important; /* Ensure input background is transparent to show parent's background */
        border: none !important; /* Remove default input border */
        outline: none !important; /* Remove default input outline */
        width: 100% !important; /* Make input fill its container */
        font-size: 1em !important;
        color: #ffffff !important; /* White text color for inputs */
    }

    /* Styling for input labels - crucial for visibility */
    div[data-testid="stWidgetLabel"] label { /* Target the label element directly within its data-testid container */
        color: #e0e0e0 !important; /* Light grey for labels for maximum contrast */
        font-weight: 500 !important; /* Slightly less bold than headers */
        margin-bottom: 5px !important;
    }


    /* Button Styling */
    .stButton>button {
        background-color: #3498db !important; /* Blue button */
        font-weight: 600 !important;
        padding: 12px 25px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }

    .stButton>button:hover {
        background-color: #2980b9 !important; /* Darker blue on hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25) !important;
    }

    .stButton>button:active, .stButton>button:focus {
        outline: none !important; /* Remove default focus outline */
        background-color: #2980b9 !important; /* Keep darker blue on active/focus */
    }

    /* Crucial: Ensure button text color is light grey in all states */
    .stButton>button p,
    .stButton>button:hover p,
    .stButton>button:active p,
    .stButton>button:focus p {
        color: #ecf0f1 !important; /* Force light grey text color */
    }


    /* Prediction result styles */
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 30px;
        text-align: center;
        font-weight: 700;
        font-size: 1.8em;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        animation: fadeIn 1s ease-out;
    }

    .prediction-box h3 {
        margin: 0;
        padding: 0;
        font-size: 1.8em;
    }

    /* Specific colors for each bucket */
    .good {
        background-color: #388e3c; /* Darker green */
        color: #e8f5e9; /* Very light green text */
        border: 2px solid #4caf50;
    }
    .satisfactory {
        background-color: #fbc02d; /* Darker yellow */
        color: #212121; /* Dark text */
        border: 2px solid #ffeb3b;
    }
    .moderate {
        background-color: #f57c00; /* Darker orange */
        color: #fff3e0; /* Very light orange text */
        border: 2px solid #ff9800;
    }
    .poor {
        background-color: #d32f2f; /* Darker red */
        color: #ffebee; /* Very light red text */
        border: 2px solid #f44336;
    }
    .very-poor {
        background-color: #b71c1c; /* Even darker red */
        color: #ffebee; /* Very light red text */
        border: 2px solid #e53935;
    }
    .severe {
        background-color: #880e4f; /* Deep maroon */
        color: #fce4ec; /* Very light pink text */
        border: 2px solid #c2185b;
    }
    .default {
        background-color: #424242; /* Dark grey */
        color: #eeeeee; /* Light grey text */
        border: 2px solid #616161;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved RandomForest model and scaler
# Ensure these files ('aqi_random_forest_best_model.pkl', 'aqi_scaler.pkl') are in the same directory as your app.py
try:
    model = joblib.load("aqi_random_forest_best_model.pkl")
    scaler = joblib.load("aqi_scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'aqi_random_forest_best_model.pkl' and 'aqi_scaler.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

# Streamlit app
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
    min_value=datetime.date(2000, 1, 1), # Set the earliest selectable date (e.g., Jan 1, 2000)
    max_value=datetime.date(2025, 12, 31) # Set the latest selectable date (e.g., Dec 31, 2025)
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
