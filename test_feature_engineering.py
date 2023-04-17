import MetaTrader5 as mt5
from feature_engineering import create_features
import pandas as pd

# Conectar ao terminal MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
else:
    print("initialize() succeeded")
    print("Terminal info:", mt5.terminal_info())
    print("Account info:", mt5.account_info())

    # Obter dados históricos
    symbol = "XAUUSD"  # ou "GOLD", se a sua corretora usar essa nomenclatura
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)

    # Desconectar do terminal
    mt5.shutdown()

    # Converter os dados históricos para um DataFrame do pandas
    data = pd.DataFrame(rates)

    if data.empty:
        print("O DataFrame está vazio.")
    else:
        print("O DataFrame contém dados.")

        # Renomear as colunas
        data.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

        # Imprimir colunas do DataFrame
        print("DataFrame columns:", data.columns)

        # Imprimir as primeiras linhas do DataFrame
        print(data.head())

        # Criar features
        data = create_features(data, open_col="open", high_col="high", low_col="low", close_col="close", volume_col="tick_volume")
