import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import date

# --- CONFIGURATION ---
FORECAST_DAYS = 30  # How many future trading days to predict
TICKER = 'PROPHET_MODEL'
# ---------------------

# 1. DATA LOADING AND PREPARATION
def create_mock_data(num_data_points=730): # ~2 years of daily data
    """Generates a DataFrame simulating stock prices with weekly seasonality."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_data_points, freq='D') 
    np.random.seed(42)
    # Base trend, strong trend, and weekly seasonality
    trend = np.arange(num_data_points) * 0.1
    noise = np.random.normal(0, 1.5, num_data_points)

    # Introduce a strong weekly seasonality component (higher prices on Friday/Monday)
    df = pd.DataFrame({'ds': dates})
    df['day_of_week'] = df['ds'].dt.dayofweek # Monday=0, Sunday=6

    # Seasonality: Higher prices on trading days (0-4), lower on weekends (5-6)
    seasonality = np.where(df['day_of_week'] < 5, 5, 0)

    df['y'] = 100 + trend + noise + seasonality

    # Filter only for business days to simulate stock data better
    df = df[df['day_of_week'] < 5] 
    return df[['ds', 'y']] # Prophet requires 'ds' and 'y' columns

# Load Data and prepare for Prophet
df_prophet = create_mock_data()

print(f"Loaded {len(df_prophet)} data points for training.")

# 2. MODEL INITIALIZATION AND TRAINING
print("Initializing and Training Prophet Model...")

# NOTE: Prophet automatically handles trend, weekly/yearly seasonality, and holidays.
model = Prophet(
    # These parameters can be tuned for better fit:
    seasonality_mode='multiplicative',
    weekly_seasonality=True,
    daily_seasonality=False # Set to True if using intra-day data
)

# Fit the model to the data
model.fit(df_prophet)

print("Model Training Complete.")

# 3. FUTURE FORECAST SETUP
# Create a DataFrame with future dates (30 business days)
future_dates = model.make_future_dataframe(periods=FORECAST_DAYS, freq='B') 
# Filter out weekends if Prophet's freq didn't already
future_dates = future_dates[future_dates['ds'].dt.dayofweek < 5] 

# 4. MAKE PREDICTIONS
forecast = model.predict(future_dates)

# 5. PLOTTING THE RESULTS
print("\nGenerating forecast plot...")

# Plot the forecast (historical data + predictions)
fig = model.plot(forecast)
plt.title(f'{TICKER} Stock Price Forecast ({FORECAST_DAYS} Days)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show() # Display plot in Jupyter

# Plot the components (trend, weekly seasonality, etc.)
fig_components = model.plot_components(forecast)
plt.show() # Display plot in Jupyter

# 6. RESULTS SUMMARY
print("\n" + "="*50)
print(f"💰 STOCK PRICE PREDICTION SUMMARY (Next {FORECAST_DAYS} Trading Days)")
# Get the last few predictions
last_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)

print("\nLast 5 Forecasted Prices (yhat is the prediction):")
print(last_predictions.to_string(index=False))
print("="*50)
