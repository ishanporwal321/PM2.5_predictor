import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta
import statistics
import pytz
import altair as alt

data = pd.DataFrame({
    'x': range(10),
    'y': np.random.randn(10)
})

# Create a simple scatter plot
chart = alt.Chart(data).mark_point().encode(
    x='x',
    y='y'
)

# Display the chart
st.altair_chart(chart, use_container_width=True)