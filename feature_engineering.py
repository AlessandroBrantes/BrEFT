import pandas as pd
import numpy as np
from ta import add_all_ta_features

def create_target(data, close_col='close', shift=-1):
    data['target'] = data[close_col].shift(shift)
    data.dropna(inplace=True)

def create_features(data, open_col, high_col, low_col, close_col, volume_col):
    data = add_all_ta_features(
        data,
        open=open_col,
        high=high_col,
        low=low_col,
        close=close_col,
        volume=volume_col,
        fillna=True
    )
    
    # Adicionar features personalizadas aqui, se necessário
    # Exemplo: data['custom_feature'] = ...

    return data
