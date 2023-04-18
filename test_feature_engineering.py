import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import create_features

def get_historical_data(symbol, timeframe, start_time, end_time):
    if not mt5.symbol_select(symbol, True):
        print(f"Não foi possível selecionar {symbol} no MetaTrader 5.")
        return pd.DataFrame()
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    
    if rates is None or len(rates) == 0:
        print(f"Não foi possível obter dados históricos para {symbol}.")
        return pd.DataFrame()
    
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    data = data[['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
    
    return data

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

print("initialize() succeeded")
print(mt5.terminal_info())
print(mt5.account_info())

symbol = "GOLD"
timeframe = mt5.TIMEFRAME_M1
start_time = datetime(2023, 4, 1)
end_time = datetime(2023, 4, 15)

data = get_historical_data(symbol, timeframe, start_time, end_time)

if data.empty:
    print("O DataFrame está vazio.")
else:
    data = create_features(data, open_col="open", high_col="high", low_col="low", close_col="close", volume_col="tick_volume")
    data['target_column'] = np.sign(data['close'].pct_change().shift(-1))
    
    # Remover linhas com valores NaN
    data.dropna(inplace=True)
    
    # Preparar os dados de entrada (features) e saída (target)
    X = data.drop(columns=["target_column"])
    y = data["target_column"]

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Avaliar o desempenho do modelo
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)

mt5.shutdown()

