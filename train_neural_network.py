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
learning_rate = 0.0005
batch_size = 32
l2_lambda = 0.001
hidden_layers = [128, 64, 32]
lookback = 50
interval = 'daily'  # Opções: '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'
symbol = 'GOLD'
alpha_vantage_interval = '60min'  # Exemplo de valores: '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'
start_date = '1998-02-20'
end_date = '2023-09-26'
api_key = 'T8Z40FYRCWEZ5OTL'  # Substitua pela sua chave de API real

# Funções auxiliares
# ...

def train_neural_network(symbol, start_date, end_date, api_key):
    # Adapte o intervalo para o formato Alpha Vantage
    interval_map = {
        '1min': '1min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '60min': '60min',
        'daily': 'daily',
        'weekly': 'weekly',
        'monthly': 'monthly'
    }

    alpha_vantage_interval = interval_map[interval]

    # Busque os dados históricos usando a API Alpha Vantage
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_intraday(symbol=symbol, interval=alpha_vantage_interval, outputsize='full')

    # Converta o índice para o formato datetime e filtre os dados para a data de início e término especificadas
    data.index = pd.to_datetime(data.index)
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # Extraia os preços 'adjusted close' e volumes
    prices = data["5. adjusted close"].values
    volumes = data["6. volume"].values

    # Normalize os preços e volumes
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    normalized_prices = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    normalized_volumes = volume_scaler.fit_transform(volumes.reshape(-1, 1)).flatten()

    # Prepare os dados de entrada e saída
    input_data = []
    output_data = []
    for i in range(len(normalized_prices) - lookback):
        input_data.append(np.hstack((normalized_prices[i:i+lookback], normalized_volumes[i:i+lookback])))
        output_data.append(normalized_prices[i+lookback])

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    # Divida os dados em conjuntos de treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # Defina a arquitetura da rede neural
    model_file = "trained_model.h5"

    # Verifique se o arquivo do modelo existe e carregue o modelo
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = Sequential()
        for i, layer_size in enumerate(hidden_layers):
            if i == 0:
                model.add(Dense(layer_size, input_dim=2 * lookback, activation="relu", kernel_regularizer=l2(l2_lambda)))
            else:
                model.add(Dense(layer_size, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(1, activation="linear", kernel_regularizer=l2(l2_lambda)))

        # Compile o modelo
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))

    # Configure a parada antecipada (early stopping)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

    # Treine o modelo com parada antecipada
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Salve o modelo treinado
    model.save(model_file)
    update_trained_model_on_github(model_file)

def predict_next_day_price(model, symbol, api_key, lookback=50):
    # Busque os dados mais recentes usando a API Alpha Vantage
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')

    # Extraia os preços 'adjusted close' e os volumes
    prices = data["5. adjusted close"].values
    volumes = data["6. volume"].values

    # Normalize os preços e os volumes
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    normalized_prices = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    normalized_volumes = volume_scaler.fit_transform(volumes.reshape(-1, 1)).flatten()

    # Prepare os dados de entrada para os últimos 50 dias
    input_data = np.hstack((normalized_prices[-lookback:], normalized_volumes[-lookback:])).reshape(1, -1)

    # Preveja o preço do próximo dia
    predicted_price = model.predict(input_data)

    # Desnormalize o preço previsto
    denormalized_price = price_scaler.inverse_transform(predicted_price)

    return denormalized_price[0][0]

if __name__ == "__main__":
    train_neural_network(symbol, start_date, end_date, api_key)

    # Carregue o modelo treinado
    model_file = "trained_model.h5"
    model = load_model(model_file)

    # Preveja o preço do próximo dia
    next_day_price = predict_next_day_price(model, symbol, api_key)

    # Mostre o preço previsto
    print(f"Preço previsto para {symbol} no próximo dia: R${next_day_price:.2f}")





