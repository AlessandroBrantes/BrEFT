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

# Criar features
data = create_features(data, open_col="open", high_col="high", low_col="low", close_col="close", volume_col="tick_volume")

# Faça algo com os dados, por exemplo, salvar em um arquivo CSV
data.to_csv("GOLD_features.csv", index=False)

