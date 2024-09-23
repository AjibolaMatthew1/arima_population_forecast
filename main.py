# Import relevant libraries
import streamlit as st
import pandas as pd

from model import population_data, forecast_population
#Streamlit interface


st.title("Okwy's Assignment: World Population Forecast App")
st.write('Select a country to view the forecasted population for the year 2025 and a comparison with historical data.')

# Dropdown to select country
countries = population_data['Country/Territory'].unique()
selected_country = st.selectbox('Select a Country', countries)

# When a country is selected, show forecast and graph
if selected_country:
    forecast, fig = forecast_population(selected_country)
    st.write(f"The forecasted population for **{selected_country}** in 2025 is: **{forecast:,.2f}**")
    st.pyplot(fig)