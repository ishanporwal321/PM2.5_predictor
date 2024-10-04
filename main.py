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
            # Commenting out the success message
            # st.success(f"Loaded model: {file}")
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
    elif 12.1 <= pm25 < 35.4:
        return "rgb(255, 255, 0)"  # Yellow
    elif 35.5 <= pm25 < 55.4:
        return "rgb(255, 126, 0)"  # Orange
    elif 55.5 <= pm25 < 150.4:
        return "rgb(255, 0, 0)"  # Red
    elif 150.5 <= pm25 < 250.4:
        return "rgb(143, 63, 151)"  # Purple
    else:
        return "rgb(126, 0, 35)"  # Maroon
    
def get_air_quality_color_in(pm25):
    if 0 <= pm25 < 30:
        return "rgb(0, 228, 0)"  # Green
    elif 31 <= pm25 < 60:
        return "rgb(255, 255, 0)"  # Yellow
    elif 61 <= pm25 < 90:
        return "rgb(255, 126, 0)"  # Orange
    elif 91 <= pm25 < 120:
        return "rgb(255, 0, 0)"  # Red
    elif 121 <= pm25 < 250:
        return "rgb(143, 63, 151)"  # Purple
    else:
        return "rgb(126, 0, 35)"  # Maroon    
    

def get_air_quality_label(pm25):
    if 0 <= pm25 < 30:
        return "Good"
    elif 31 <= pm25 < 60:
        return "Moderate"
    elif 61 <= pm25 < 90:
        return "Unhealthy for Sensitive Groups"
    elif 91 <= pm25 < 120:
        return "Unhealthy"
    elif 121 <= pm25 < 250:
        return "Very Unhealthy"
    else:
        return "Hazardous"
    
def get_air_quality_color_in(pm25):
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

def get_air_quality_label_in(pm25):
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
st.title("PM2.5 and AQI for Indore City")

# Fetch data for today and yesterday
today = datetime.now(pytz.UTC)
yesterday = today - timedelta(days=1)

# st.subheader("Weather Data Fetched")
# st.write("Today's data:")
today_data = fetch_weather_data(today)
# if today_data:
#     st.json(today_data)
# else:
#     st.error("Failed to fetch today's weather data")

# st.write("Yesterday's data:")
yesterday_data = fetch_weather_data(yesterday)
# if yesterday_data:
#     st.json(yesterday_data)
# else:
#     st.error("Failed to fetch yesterday's weather data")

# Other imports and code...

def get_aqi_message(aqi):
    """Return a message based on the AQI value."""
    if aqi < 20:
        return ("Excellent", "The air quality is ideal for most individuals; enjoy your normal outdoor activities.")
    elif aqi < 50:
        return ("Fair", "The air quality is generally acceptable for most individuals. However, sensitive groups may experience minor to moderate symptoms from long-term exposure.")
    elif aqi < 100:
        return ("Poor", "The air has reached a high level of pollution and is unhealthy for sensitive groups. Reduce time spent outside if you are feeling symptoms such as difficulty breathing or throat irritation.")
    elif aqi < 150:
        return ("Unhealthy", "Health effects can be immediately felt by sensitive groups. Healthy individuals may experience difficulty breathing and throat irritation with prolonged exposure. Limit outdoor activity.")
    elif aqi < 250:
        return ("Very Unhealthy", "Health effects will be immediately felt by sensitive groups and should avoid outdoor activity. Healthy individuals are likely to experience difficulty breathing and throat irritation; consider staying indoors and rescheduling outdoor activities.")
    else:
        return ("Dangerous", "Any exposure to the air, even for a few minutes, can lead to serious health effects on everybody. Avoid outdoor activities.")
    

def calculate_aqi(pm25):
    """Calculate AQI based on PM2.5 concentration."""
    if pm25 < 12:
        return pm25 * 50 / 12  # Good
    elif pm25 < 35.5:
        return 50 + (pm25 - 12) * 50 / (35.5 - 12)  # Moderate
    elif pm25 < 55.5:
        return 100 + (pm25 - 35.5) * 100 / (55.5 - 35.5)  # Unhealthy for Sensitive Groups
    elif pm25 < 150.5:
        return 150 + (pm25 - 55.5) * 100 / (150.5 - 55.5)  # Unhealthy
    elif pm25 < 250.5:
        return 200 + (pm25 - 150.5) * 100 / (250.5 - 150.5)  # Very Unhealthy
    else:
        return 300 + (pm25 - 250.5) * 100 / (500 - 250.5)  # Hazardous
    
def calculate_aqi_in(pm25):
    """Calculate AQI based on PM2.5 concentration."""
    if pm25 <= 30:
        return pm25 * 50 / 30  # Good
    elif pm25 <= 60:
        return 50 + (pm25 - 30) * 50 / (60 - 30)  # Moderate
    elif pm25 <= 90:
        return 100 + (pm25 - 60) * 100 / (90 - 60)  # Unhealthy for Sensitive Groups
    elif pm25 <= 120:
        return 200 + (pm25 - 90) * 100 / (120 - 90)  # Unhealthy
    elif pm25 < 250:
        return 300 + (pm25 - 120) * 100 / (250 - 120)  # Very Unhealthy
    else:
        return 400 + (pm25 - 250) * 100 / (500 - 250)

def fetch_forecast_data():
    api_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_APPID}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('list'):
            daily_pm25 = {}
            for item in data['list']:
                # Convert the timestamp to a date
                timestamp = item['dt']
                date = datetime.fromtimestamp(timestamp, pytz.UTC).date()
                
                # Get the PM2.5 value
                pm25_value = item['components']['pm2_5']
                
                # Group PM2.5 values by date
                if date not in daily_pm25:
                    daily_pm25[date] = []
                daily_pm25[date].append(pm25_value)
            
            # Calculate the average PM2.5 for each day
            average_pm25 = {date: statistics.mean(values) for date, values in daily_pm25.items()}
            return average_pm25
        else:
            st.error("No forecast data available in the response")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch forecast data: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

# In your Streamlit app logic, after making predictions
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

        # Fetch forecast data
        forecast_data = fetch_forecast_data()

        # Prepare data for comparison
        if forecast_data:
            # Get the dates for the next 5 days
            forecast_dates = [(today + timedelta(days=i)).date() for i in range(5)]
            forecast_pm25 = [forecast_data.get(date, None) for date in forecast_dates]

            # Display results            
            # Prepare data for the chart
            colors = [get_air_quality_color(pred) for pred in predictions]
            
            # Create Plotly figure
            fig = go.Figure()

            # Add predicted values
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted PM2.5 by IIT Indore',
                line=dict(color='rgb(0, 0, 0)', width=2),
                marker=dict(color=colors, size=10),
                text=[f"{pred:.2f} µg/m³" for pred in predictions],
                hoverinfo='text+x'
            ))

            # Add forecast values
            if forecast_pm25:
                forecast_colors = [get_air_quality_color(forecast) for forecast in forecast_pm25]
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_pm25,
                    mode='lines+markers',
                    name='Forecast PM2.5 by Open Weather',
                    line=dict(color='rgb(255, 0, 0)', width=2, dash='dash'),
                    marker=dict(color=forecast_colors, size=10),
                    text=[f"{forecast:.2f} µg/m³" for forecast in forecast_pm25],
                    hoverinfo='text+x'
                ))

            fig.update_layout(
                title='PM2.5 Predictions by IIT Indore vs Forecast by Open Weather',
                xaxis_title='Date',
                yaxis_title='PM2.5 (µg/m³)',
                height=400
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # Display predictions with cards
            st.write("Detailed Predictions:")

            # First row - Full-width card
            with st.container():
                # Calculate AQI from predicted PM2.5
                predicted_aqi = calculate_aqi(predictions[0])  # Assuming predictions[0] is for the first date
                color = get_air_quality_color(predictions[0])
                label = get_air_quality_label(predictions[0])
                predicted_aqi_in = calculate_aqi_in(predictions[0])  # Assuming predictions[0] is for the first date
                color_in = get_air_quality_color_in(predictions[0])
                label_in = get_air_quality_label(predictions[0])                
                
                # Get AQI message
                aqi_label, aqi_message = get_aqi_message(predicted_aqi)
                
                # Use HTML and CSS to create a full-width card
                html_content = f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    overflow: hidden;
                    height: 400px;
                    width: 100%;
                    margin-bottom: 20px;  /* Add margin to create space below the card */
                ">
                    <div style="
                        background-color: {color};
                        height: 20%;
                    "></div>
                    <div style="
                        padding: 20px;  /* Increased padding for better spacing */
                        text-align: center;
                        font-family: Arial, sans-serif;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="font-size: 18px;">{label}</strong>
                            <strong style="font-size: 18px;">{forecast_dates[0]}</strong>
                        </div>
                        <div style="font-size: 24px; margin: 10px 0;">{predictions[0]:.2f} µg/m³</div>
                        <div style="
                            background-color: {'#5cb85c' if predicted_aqi_in < 50 else '#f0ad4e' if predicted_aqi < 100 else '#d9534f'};
                            padding: 10px;
                            border-radius: 5px;
                            display: inline-block;
                            margin-top: 10px;
                        ">
                            <i class="fas fa-wind" style="font-size: 24px; color: white;"></i><br>
                            <strong style="font-size: 15px; color: white;">Indian AQI</strong><br>                          
                            <strong style="font-size: 20px; color: white;">AQI: {predicted_aqi_in:.2f}</strong>
                        </div>
                            <div style="
                                background-color: {'#5cb85c' if predicted_aqi < 50 else '#f0ad4e' if predicted_aqi < 100 else '#d9534f'};
                                padding: 10px;
                                border-radius: 5px;
                                display: inline-block;
                                margin-top: 10px;
                            ">
                                <i class="fas fa-wind" style="font-size: 24px; color: white;"></i><br>
                                <strong style="font-size: 15px; color: white;">US EPA AQI</strong><br>                          
                                <strong style="font-size: 20px; color: white;">AQI: {predicted_aqi:.2f}</strong>
                            </div>                        
                        <div style="margin-top: 10px; font-size: 16px; color: grey;">
                            <strong>{aqi_label}</strong>: {aqi_message}
                        </div>
                    </div>
                </div>
                """
                st.markdown(html_content, unsafe_allow_html=True)

            # Second row - Two columns with two cards each
            cols = st.columns(2)

            for i in range(1, 5):  # Start from the second prediction
                with cols[i % 2]:  # Alternate between the two columns
                    predicted_aqi = calculate_aqi(predictions[i])
                    color = get_air_quality_color(predictions[i])
                    label = get_air_quality_label(predictions[i])
                    predicted_aqi_in = calculate_aqi_in(predictions[i])
                    color_in = get_air_quality_color_in(predictions[i])
                    label_in = get_air_quality_label(predictions[i])                    
                    
                    # Get AQI message
                    aqi_label, aqi_message = get_aqi_message(predicted_aqi)
                    
                    html_content = f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        overflow: hidden;
                        height: 500px;
                        margin-bottom: 20px;  /* Add margin to create space below the card */
                    ">
                        <div style="
                            background-color: {color};
                            height: 20%;
                        "></div>
                        <div style="
                            padding: 20px;  /* Increased padding for better spacing */
                            text-align: center;
                            font-family: Arial, sans-serif;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-size: 18px;">{label}</strong>
                                <strong style="font-size: 18px;">{forecast_dates[i]}</strong>
                            </div>
                            <div style="font-size: 24px; margin: 10px 0;">{predictions[i]:.2f} µg/m³</div>
                            <div style="
                                background-color: {'#5cb85c' if predicted_aqi_in < 50 else '#f0ad4e' if predicted_aqi < 100 else '#d9534f'};
                                padding: 10px;
                                border-radius: 5px;
                                display: inline-block;
                                margin-top: 10px;
                            ">
                                <i class="fas fa-wind" style="font-size: 24px; color: white;"></i><br>
                                <strong style="font-size: 15px; color: white;">Indian AQI</strong><br>                          
                                <strong style="font-size: 20px; color: white;">AQI: {predicted_aqi_in:.2f}</strong>
                            </div>
                            <div style="
                                background-color: {'#5cb85c' if predicted_aqi < 50 else '#f0ad4e' if predicted_aqi < 100 else '#d9534f'};
                                padding: 10px;
                                border-radius: 5px;
                                display: inline-block;
                                margin-top: 10px;
                            ">
                                <i class="fas fa-wind" style="font-size: 24px; color: white;"></i><br>
                                <strong style="font-size: 15px; color: white;">US EPA AQI</strong><br>                          
                                <strong style="font-size: 20px; color: white;">AQI: {predicted_aqi:.2f}</strong>
                            </div>                            
                            <div style="margin-top: 10px; font-size: 16px; color: grey;">
                                <strong>{aqi_label}</strong>: {aqi_message}
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(html_content, unsafe_allow_html=True)
            



        else:
            st.error("Failed to fetch forecast data. Cannot proceed with comparison.")

    else:
        st.error("Failed to fetch PM2.5 data. Cannot proceed with predictions.")

else:
    st.error("Failed to fetch complete data. Cannot proceed with predictions.")