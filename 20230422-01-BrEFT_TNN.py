import numpy as np
import pandas as pd
import os
import subprocess
import yfinance as yf
from datetime import timedelta
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
end_date = None  # Altere a data final para usar um intervalo de tempo diferente para análise
learning_rate = 0.0005  # Altere a taxa de aprendizado para ajustar a velocidade de convergência do modelo
batch_size = 32  # Altere o tamanho do lote para ajustar a quantidade de exemplos usados em cada atualização do modelo
l2_lambda = 0.001  # Altere o fator de regularização L2 para ajustar a penalidade aplicada aos pesos do modelo
hidden_layers = [128, 64, 32]  # Altere o número e o tamanho das camadas ocultas para ajustar a arquitetura do modelo
lookback = 25  # Altere o número de dias analisados para ajustar a quantidade de informações temporais usadas pelo modelo
interval = 'daily'  # Altere o intervalo de tempo dos dados para análise (opções: '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')

# Definir o caminho do diretório local para salvar o arquivo do modelo
seu_diretorio_local = 'C:\\Users\\Alessandro Brantes\\BrEFT'
trained_model_path = os.path.join(seu_diretorio_local, f'trained_model_{symbol}.h5')

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

def train_neural_network(symbol, start_date, end_date, lookback, learning_rate, batch_size, l2_lambda, hidden_layers):
    try:
        # Fetch the historical data using Yahoo Finance API
        data = yf.download(symbol, start=start_date, end=end_date)

        # Convert the index to datetime format and filter the data for the specified start_date and end_date
        data.index = pd.to_datetime(data.index)
        if end_date is None:
            data = data[data.index >= start_date]
        else:
            data = data[(data.index >= start_date) & (data.index <= end_date)]

        # Prepare the input and output data
        input_data, output_data, price_scaler, volume_scaler = prepare_data(data, lookback, 'Adj Close', 'Volume')

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

def prepare_data(data, lookback, price_col_name='5. adjusted close', volume_col_name='6. volume'):
    prices = data[price_col_name].values
    volumes = data[volume_col_name].values

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

def predict_next_day_price_yf(model, symbol, lookback):
    # Fetch the latest data using yfinance
    data = yf.download(symbol, period="60d")

    # Prepare the input data
    input_data, _, price_scaler, volume_scaler = prepare_data(data, lookback, 'Adj Close', 'Volume')

    # Predict the next day price
    predicted_price = model.predict(input_data[-1].reshape(1, -1))

    # Denormalize the predicted price
    denormalized_price = price_scaler.inverse_transform(predicted_price)

    return denormalized_price[0][0]

if __name__ == "__main__":
    train_neural_network(symbol, start_date, end_date, lookback, learning_rate, batch_size, l2_lambda, hidden_layers)

    # Load the trained model
    model = load_model(trained_model_path)

    # Predict the next day price using the new function predict_next_day_price_yf
    next_day_price = predict_next_day_price_yf(model, symbol, lookback)

    # Obter a última data do conjunto de dados usando yfinance
    data = yf.download(symbol, period="60d")
    data.index = pd.to_datetime(data.index)
    last_date = data.index[-1]

    # Calcular a data da previsão do preço para o próximo dia
    next_day_date = last_date + timedelta(days=1)

    # Print the start and end dates
    print(f"Data de início: {start_date}")
    if end_date is None:
        print(f"Data final: {last_date.strftime('%Y-%m-%d')}")
    else:
        print(f"Data final: {end_date}")

    # Print the predicted price
    print(f"Preço previsto para {symbol} em {next_day_date.strftime('%Y-%m-%d')}: ${next_day_price:.2f}")