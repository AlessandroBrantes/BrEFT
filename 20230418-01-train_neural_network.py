import numpy as np
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Define the neural network training process
def train_neural_network(symbol, start_date, end_date, api_key):
    lookback = 50

    # Fetch the historical data using Alpha Vantage API
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')

    # Convert the index to datetime format and filter the data for the specified start_date and end_date
    data.index = pd.to_datetime(data.index)
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # Extract the 'adjusted close' prices
    prices = data["5. adjusted close"].values

    # Normalize the prices
    scaler = MinMaxScaler()
    normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # Prepare the input and output data
    input_data = []
    output_data = []
    for i in range(len(normalized_prices) - lookback):
        input_data.append(normalized_prices[i:i+lookback])
        output_data.append(normalized_prices[i+lookback])

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # Define the neural network architecture
    model_file = "trained_model.h5"

    # Check if the model file exists and load the model
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = Sequential()
        model.add(Dense(64, input_dim=lookback, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="linear"))

        # Compile the model
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))

    # Configure early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

    # Train the model with early stopping
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Save the trained model
    model.save(model_file)


if __name__ == "__main__":
    symbol = "GOLD"
    start_date = "2018-01-01"
    end_date = "2023-04-17"
    api_key = "T8Z40FYRCWEZ5OTL"  # Replace with your actual API key
    train_neural_network(symbol, start_date, end_date, api_key)



