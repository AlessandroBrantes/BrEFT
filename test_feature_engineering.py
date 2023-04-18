import pandas as pd
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

def test_feature_engineering(asset):
    data = pd.read_csv(f"{asset}_M1.csv")
    data = create_features(data)
    data = create_target(data)
    
    print(data.head())
    
    X = data.drop(columns=["target"])
    y = data["target"]

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    print("Head of X:\n", X.head())
    print("Head of y:\n", y.head())

if __name__ == "__main__":
    asset = "GOLD"
    test_feature_engineering(asset)




