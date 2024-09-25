import os
import streamlit as st
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import pandas as pd
import requests
from datetime import datetime, timedelta
import statistics
import pytz
import plotly.graph_objects as go
import json

# Load environment variables
load_dotenv()

API_KEY = os.getenv('API_KEY')
OPENWEATHER_APPID = os.getenv('OPENWEATHER_APPID')

# Constants
LAT = "22.724"
LON = "75.857"


def create_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Function to load or create models
def load_or_create_models():
    model_files = ['best_cnn_model.keras', 'best_lstm_model.keras', 'best_gru_model.keras']
    models = []
    
    for file in model_files:
        try:
            model = tf.keras.models.load_model(file)
            st.success(f"Loaded model: {file}")
            models.append(model)
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
            st.warning(f"Creating new model to replace {file}")
            models.append(create_model((9, 1)))  # Adjust input shape if necessary
    
    return models

# Load or create models
best_cnn_lstm_model, best_cnn_rnn_model, best_cnn_gru_model = load_or_create_models()

def convert_to_ms(speed):
    return speed * 0.44704 if speed > 100 else speed

def fetch_weather_data(date):
    ist = pytz.timezone('Asia/Kolkata')
    utc = pytz.UTC
    
    end_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=1)
    
    start_timestamp = int(start_date.astimezone(utc).timestamp())
    end_timestamp = int(end_date.astimezone(utc).timestamp())
    
    api_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={LAT}&lon={LON}&type=hour&start={start_timestamp}&end={end_timestamp}&appid={OPENWEATHER_APPID}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('list'):
            temps = [item['main']['temp'] for item in data['list']]
            wind_speeds = [convert_to_ms(item['wind']['speed']) for item in data['list']]
            precipitation = sum(item.get('rain', {}).get('1h', 0) for item in data['list'])
            
            weather_data = {
                'Tmax': max(temps) - 273.15,
                'Tmin': min(temps) - 273.15,
                'P': precipitation,
                'w': statistics.mean(wind_speeds)
            }
            return weather_data
        else:
            st.error("No weather data available in the response")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch weather data: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def fetch_pm25(date=None):
    if date is None:
        date = datetime.now(pytz.UTC)  # Default to today if no date is provided

    # Convert the date to timestamps
    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    end_date = date.replace(hour=0, minute=0, second=0, microsecond=0)

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    api_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_timestamp}&end={end_timestamp}&appid={OPENWEATHER_APPID}"
    
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if data['list']:
            pm25_values = [item['components']['pm2_5'] for item in data['list']]
            avg_pm25 = statistics.mean(pm25_values)
            return avg_pm25
        else:
            st.error("No PM2.5 data available for the requested period")
            return None
    else:
        st.error(f"Failed to fetch PM2.5 data: {response.status_code}")
        return None

def create_input_sequence(PM25_prev, P, P_prev, Tmin, Tmin_prev, Tmax, Tmax_prev, w, w_prev, n_steps_in=1):
    sequence = np.array([[PM25_prev, P, P_prev, Tmin, Tmin_prev, Tmax, Tmax_prev, w, w_prev]])
    sequence = sequence.reshape((sequence.shape[0], sequence.shape[1], 1))
    return sequence

def predict_pm25(sequence, days=5):
    predictions = []
    current_sequence = sequence.copy()
    
    for _ in range(days):
        cnn_lstm_pred = best_cnn_lstm_model.predict(current_sequence)[0, 0]
        cnn_rnn_pred = best_cnn_rnn_model.predict(current_sequence)[0, 0]
        cnn_gru_pred = best_cnn_gru_model.predict(current_sequence)[0, 0]
        
        prediction = (cnn_lstm_pred + cnn_rnn_pred + cnn_gru_pred) / 3
        predictions.append(prediction)
        
        current_sequence = np.roll(current_sequence, shift=-1, axis=1)
        current_sequence[0, -1, 0] = prediction
    
    return predictions

def get_air_quality_color(pm25):
    if 0 <= pm25 < 12:
        return "rgb(0, 228, 0)"  # Green
    elif 12 <= pm25 < 35.5:
        return "rgb(255, 255, 0)"  # Yellow
    elif 35.5 <= pm25 < 55.5:
        return "rgb(255, 126, 0)"  # Orange
    elif 55.5 <= pm25 < 150.5:
        return "rgb(255, 0, 0)"  # Red
    elif 150.5 <= pm25 < 250.5:
        return "rgb(143, 63, 151)"  # Purple
    else:
        return "rgb(126, 0, 35)"  # Maroon

def get_air_quality_label(pm25):
    if 0 <= pm25 < 12:
        return "Good"
    elif 12 <= pm25 < 35.5:
        return "Moderate"
    elif 35.5 <= pm25 < 55.5:
        return "Unhealthy for Sensitive Groups"
    elif 55.5 <= pm25 < 150.5:
        return "Unhealthy"
    elif 150.5 <= pm25 < 250.5:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Streamlit app
st.title("PM2.5 Prediction for Today and Next 4 Days")

# Fetch data for today and yesterday
today = datetime.now(pytz.UTC)
yesterday = today - timedelta(days=1)

st.subheader("Weather Data Fetched")
st.write("Today's data:")
today_data = fetch_weather_data(today)
if today_data:
    st.json(today_data)
else:
    st.error("Failed to fetch today's weather data")

st.write("Yesterday's data:")
yesterday_data = fetch_weather_data(yesterday)
if yesterday_data:
    st.json(yesterday_data)
else:
    st.error("Failed to fetch yesterday's weather data")

if today_data and yesterday_data:
    # Fetch PM2.5 data
    today_pm25 = fetch_pm25(today)
    yesterday_pm25 = fetch_pm25(yesterday)

    if today_pm25 is not None and yesterday_pm25 is not None:
        # Create input sequence
        sequence = create_input_sequence(
            PM25_prev=yesterday_pm25,
            P=today_data['P'],
            P_prev=yesterday_data['P'],
            Tmin=today_data['Tmin'],
            Tmin_prev=yesterday_data['Tmin'],
            Tmax=today_data['Tmax'],
            Tmax_prev=yesterday_data['Tmax'],
            w=today_data['w'],
            w_prev=yesterday_data['w']
        )

        # Make predictions
        predictions = predict_pm25(sequence)

        # Display results
        st.subheader("PM2.5 Predictions")
        
        # Prepare data for the chart
        dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)]
        colors = [get_air_quality_color(pred) for pred in predictions]
        
        # Create Plotly figure
        fig = go.Figure(data=go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            line=dict(color='rgb(0, 0, 0)', width=2),
            marker=dict(color=colors, size=10),
            text=[f"{pred:.2f} µg/m³" for pred in predictions],
            hoverinfo='text+x'
        ))
        
        fig.update_layout(
            title='PM2.5 Predictions for the Next 5 Days',
            xaxis_title='Date',
            yaxis_title='PM2.5 (µg/m³)',
            height=400
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display predictions with cards
        st.write("Detailed Predictions:")
        cols = st.columns(5)  # Create 5 columns for the 5 predictions
        
        for i, (date, pred) in enumerate(zip(dates, predictions)):
            color = get_air_quality_color(pred)
            label = get_air_quality_label(pred)
            
            with cols[i]:
                # Create a card-like container
                with st.container():
                    # Use HTML and CSS to create a split-color card
                    html_content = f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        overflow: hidden;
                        height: 180px;
                    ">
                        <div style="
                            background-color: {color};
                            height: 50%;
                        "></div>
                        <div style="
                            padding: 10px;
                            text-align: center;
                        ">
                            <strong>{date}</strong><br>
                            {pred:.2f} µg/m³<br>
                            {label}
                        </div>
                    </div>
                    """
                    st.markdown(html_content, unsafe_allow_html=True)

    else:
        st.error("Failed to fetch PM2.5 data. Cannot proceed with predictions.")

else:
    st.error("Failed to fetch complete data. Cannot proceed with predictions.")

# Display input data
st.subheader("Input Data")
if today_data and yesterday_data:
    st.write("Today's data:", today_data)
    st.write("Yesterday's data:", yesterday_data)
    st.write(f"Yesterday's PM2.5: {yesterday_pm25:.2f}")
else:
    st.write("No input data available")