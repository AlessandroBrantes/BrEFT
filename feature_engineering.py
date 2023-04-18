import pandas as pd
import numpy as np
from ta import add_all_ta_features
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def create_features(df):
    df['time'] = pd.to_datetime(df['time'], format="%Y.%m.%d %H:%M:%S")
    df.set_index('time', inplace=True)
    
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="tick_volume"
    )

    return df

def create_target(df, shift=-1):
    df['target'] = df['close'].shift(shift)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    data = pd.read_csv("GOLD_M1.csv")
    data = create_features(data)
    data = create_target(data)
    data.to_csv("GOLD_M1_processed.csv")
