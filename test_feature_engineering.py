import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from feature_engineering import create_features
from sklearn.model_selection import train_test_split

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
    data_with_features = create_features(data, open_col="open", high_col="high", low_col="low", close_col="close", volume_col="tick_volume")
    
    # Imprimir as primeiras linhas dos dados com as features técnicas adicionadas
    print(data_with_features.head())

    # Dividir os dados em conjuntos de treinamento e teste
    target_variable = "target"
    X = data_with_features.drop(columns=[target_variable])
    y = data_with_features[target_variable]

    # Adicionar mensagens de depuração
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Head of X:")
    print(X.head())
    print("Head of y:")
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mt5.shutdown()


