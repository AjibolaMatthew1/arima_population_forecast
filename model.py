# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


# Load the population data
population_data = pd.read_csv('world_population.csv')

# Function to perform the ADF test and return the p-value
def adf_test(series):
    result = adfuller(series)
    return result[1]  # Return p-value from ADF test

# Function to perform differencing if needed based on ADF test result
def make_stationary(data):
    d = 0  # Differencing level
    adf_pvalue = adf_test(data)
    
    # Perform differencing while p-value is above 0.05 (non-stationary)
    while adf_pvalue > 0.05:
        data = np.diff(data)  # Differencing the data
        d += 1
        adf_pvalue = adf_test(data)
    
    # Convert to a pandas Series to drop NaN values resulting from differencing
    data = pd.Series(data).dropna()

    return data, d

# Function to forecast population using auto_arima for a specific country
def forecast_population(country_name, forecast_year=2025):
    # Filter the data for the selected country
    country_data = population_data[population_data['Country/Territory'] == country_name]
    
    # Extract the relevant years and population data from 1970 to 2022
    years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]
    populations = [
        country_data['1970 Population'].values[0],
        country_data['1980 Population'].values[0],
        country_data['1990 Population'].values[0],
        country_data['2000 Population'].values[0],
        country_data['2010 Population'].values[0],
        country_data['2015 Population'].values[0],
        country_data['2020 Population'].values[0],
        country_data['2022 Population'].values[0]
    ]

    # Check stationarity and apply differencing if necessary
    stationary_data, d_value = make_stationary(populations)

    # Use auto_arima to find the best p, d, q values
    model_auto = auto_arima(stationary_data, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False, trace=True, stepwise=True)

    #Fit the model
    model_auto.fit(populations)

    # model = ARIMA(populations, order=(1,0 , 0))
    # model_fit = model.fit(populations)

    # Forecast population for the given year
    # Check if the model is ARIMA(0, 0, 0)
    if model_auto.order == (0, 0, 0):
        # Fallback if the model is ARIMA(0, 0, 0), just use last known population
        forecast = populations[-1]
        forecasted_values = np.array([populations[-1]]*3)
        #st.write(f"Auto ARIMA selected ARIMA(0, 0, 0), which may not provide meaningful forecasts. Returning the last known population value.")
    else:
        # Forecast population for the given year
        forecast_steps = forecast_year - years[-1]
        forecasted_values = model_auto.predict(n_periods=forecast_steps)
        print(forecasted_values)
        # Check if there are forecasted values, if not, fallback to last known population
        #forecast = forecasted_values.iloc[-1]
        forecast = forecasted_values[-1]
    # forecast_steps = forecast_year - years[-1]
    # forecasted_values = model_fit.predict(n_periods=forecast_steps)
    # #     # Check if there are forecasted values, if not, fallback to last known population
    # #forecast = forecasted_values.iloc[-1]
    # forecast = forecasted_values[-1]
    # print(forecasted_values)


    # Plot the historical and forecasted population
    future_years = [2023, 2024, 2025]
    years_extended = years + future_years

    # Extend populations to include forecasted values
    extended_populations = populations + list(forecasted_values)

    # Plot the historical and forecasted population
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot historical population (up to 2022)
    ax.plot(years, populations, marker='o', label='Historical Population', color='blue')

    # Plot forecasted population (2023 to 2025) and connect to 2022
    ax.plot([years[-1]] + future_years, [populations[-1]] + list(forecasted_values), marker='o', color='green', label='Forecasted Population (2023-2025)')

    # Set title and labels
    ax.set_title(f'Population Forecast for {country_name}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Population')

    # Set x-ticks and rotate them vertically
    ax.set_xticks(years_extended)  # Set x-ticks to include future years
    plt.xticks(rotation=90)  # Rotate the x-axis labels vertically

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    plt.show()
    
    return forecast , fig

forecast_population('Nigeria')
