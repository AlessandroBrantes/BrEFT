import numpy as np
import pandas as pd
import os
import subprocess
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Parâmetros ajustáveis
symbol = 'MSFT'  # Altere o símbolo da empresa para analisar ações de outras empresas
start_date = '1998-02-20'  # Altere a data de início para usar um intervalo de tempo diferente para análise
end_date = '2023-09-26'  # Altere a data final para usar um intervalo de tempo diferente para análise
api_key = 'T8Z40FYRCWEZ5OTL'  # Insira sua chave de API do Alpha Vantage
learning_rate = 0.0005  # Altere a taxa de aprendizado para ajustar a velocidade de convergência do modelo
batch_size = 32  # Altere o tamanho do lote para ajustar a quantidade de exemplos usados em cada atualização do modelo
l2_lambda = 0.001  # Altere o fator de regularização L2 para ajustar a penalidade aplicada aos pesos do modelo
hidden_layers = [128, 64, 32]  # Altere o número e o tamanho das camadas ocultas para ajustar a arquitetura do modelo
lookback = 25  # Altere o número de dias analisados para ajustar a quantidade de informações temporais usadas pelo modelo
interval = 'daily'  # Altere o intervalo de tempo dos dados para análise (opções: '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')
alpha_vantage_interval = '60min'  # Altere o intervalo de tempo usado para obter dados do Alpha Vantage (valores de exemplo: '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')

# Definir o caminho do diretório local para salvar o arquivo do modelo
seu_diretorio_local = 'C:\\Users\\Alessandro Brantes\\BrEFT'
trained_model_path = os.path.join(seu_diretorio_local, 'trained_model_MSFT.h5')

def git_commit_push(file_name, commit_message):
    try:
        subprocess.check_output(['git', 'add', file_name])
        subprocess.check_output(['git', 'commit', '-m', commit_message])
        subprocess.check_output(['git', 'push', 'origin', 'main'])
        print(f"Arquivo {file_name} enviado com sucesso ao GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao enviar arquivo {file_name} ao GitHub: {e.output.decode()}")

def update_trained_model_on_github(model_file):
    commit_message = f"Atualização automática do arquivo {model_file}"
    git_commit_push(model_file, commit_message)

def train_neural_network(symbol, start_date, end_date, api_key, lookback, learning_rate, batch_size, l2_lambda, hidden_layers):
    try:
        # Fetch the historical data using Alpha Vantage API
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')

        # Convert the index to datetime format and filter the data for the specified start_date and end_date
        data.index = pd.to_datetime(data.index)
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        # Prepare the input and output data
        input_data, output_data, price_scaler, volume_scaler = prepare_data(data, lookback)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

        # Define the neural network architecture
        model_file = f"trained_model.h5_{symbol}"

        # Check if the model file exists and load the model
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            model = Sequential()
            for i, layer_size in enumerate(hidden_layers):
                if i == 0:
                    model.add(Dense(layer_size, input_dim=lookback * 2, activation="relu", kernel_regularizer=l2(l2_lambda)))
                else:
                    model.add(Dense(layer_size, activation="relu", kernel_regularizer=l2(l2_lambda)))
            model.add(Dense(1, activation="linear", kernel_regularizer=l2(l2_lambda)))

            # Compile the model
            model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))

        # Configure early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

        # Train the model with early stopping
        model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Save the trained model
        model.save(trained_model_path)
        update_trained_model_on_github(trained_model_path)

    except ValueError as ve:
        print(f"Erro ao obter dados para o símbolo {symbol}: {ve}")
        return

def prepare_data(data, lookback):
    prices = data["5. adjusted close"].values
    volumes = data["6. volume"].values

    # Normalize the prices and volumes
    price_scaler = MinMaxScaler()
    normalized_prices = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    volume_scaler = MinMaxScaler()
    normalized_volumes = volume_scaler.fit_transform(volumes.reshape(-1, 1)).flatten()

    # Prepare the input and output data
    input_data = []
    output_data = []
    for i in range(len(normalized_prices) - lookback):
        input_data.append(np.concatenate((normalized_prices[i:i+lookback], normalized_volumes[i:i+lookback])).flatten())
        output_data.append(normalized_prices[i+lookback])

    return np.array(input_data), np.array(output_data), price_scaler, volume_scaler

def predict_next_day_price(model, symbol, api_key, lookback=50):
    # Fetch the latest data using Alpha Vantage API
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')

    # Prepare the input data
    input_data, _, price_scaler, volume_scaler = prepare_data(data, lookback)

    # Predict the next day price
    predicted_price = model.predict(input_data[-1].reshape(1, -1))

    # Denormalize the predicted price
    denormalized_price = price_scaler.inverse_transform(predicted_price)

    return denormalized_price[0][0]

if __name__ == "__main__":
    train_neural_network(symbol, start_date, end_date, api_key, lookback, learning_rate, batch_size, l2_lambda, hidden_layers)
    
    # Load the trained model
    model_file = f"trained_model.h5_{symbol}"
    model = load_model(model_file)

    # Predict the next day price
    next_day_price = predict_next_day_price(model, symbol, api_key, lookback)

    # Print the predicted price
    print(f"Predicted price for {symbol} on the next day: ${next_day_price:.2f}")