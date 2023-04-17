import pandas as pd
from ta import add_all_ta_features

def create_features(data):
    # Calcular indicadores técnicos
    data = add_all_ta_features(data, open="open", high="high", low="low", close="close", volume="tick_volume")
    return data
