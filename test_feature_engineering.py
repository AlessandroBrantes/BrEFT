import MetaTrader5 as mt5
from feature_engineering import create_features
import pandas as pd

# Conectar ao terminal MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Obter dados históricos
symbol = "GOLD"  # ou "XAUUSD", se a sua corretora usar essa nomenclatura
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)

# Desconectar do terminal
mt5.shutdown()

# Converter os dados históricos para um DataFrame do pandas
data = pd.DataFrame(rates)

# Renomear as colunas
data.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

# Imprimir colunas do DataFrame
print("DataFrame columns:", data.columns)

# Criar features
data = create_features(data)
