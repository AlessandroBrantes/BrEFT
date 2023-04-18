import pandas as pd
from feature_engineering import create_features, create_target

def test_feature_engineering():
    data = pd.read_csv("GOLD_M1.csv")
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
    test_feature_engineering()



