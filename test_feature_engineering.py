import pandas as pd
from feature_engineering import create_features
import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

symbol = "GOLD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)

mt5.shutdown()

data = pd.DataFrame(rates)
data = create_features(data)

print("Features criadas com sucesso:")
print(data.head())
