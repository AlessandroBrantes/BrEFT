import MetaTrader5 as mt5

# Conectar ao terminal MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Obter dados históricos
symbol = "GOLD"  # ou "XAUUSD", se a sua corretora usar essa nomenclatura
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)

# Desconectar do terminal
mt5.shutdown()
