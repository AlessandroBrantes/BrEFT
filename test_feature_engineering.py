import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from feature_engineering import create_features, create_target

symbol = "GOLD"
timezone_offset = -3
timeframe = mt5.TIMEFRAME_M1
start_time = datetime(2023, 4, 3)
end_time = datetime(2023, 4, 4)

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    quit()

rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)

if data.empty:
    print("O DataFrame está vazio.")
else:
    data_with_features = create_features(data, open_col="open", high_col="high", low_col="low", close_col="close", volume_col="tick_volume")
    
    # Adicionar a coluna 'target'
    create_target(data_with_features, close_col='close', shift=-1)
    
    # Imprimir as primeiras linhas dos dados com as features técnicas adicionadas
    print(data_with_features.head())

    # Dividir os dados em conjuntos de treinamento e teste
    target_variable = "target"
    X = data_with_features.drop(columns=[target_variable])
    y = data_with_features[target_variable]

    # Adicionar mensagens de depuração
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Head of X:")
    print(X.head())
    print("Head of y:")
    print(y.head())

    # Etapa 3: Seleção e treinamento do modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Avaliar o desempenho do modelo
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)

mt5.shutdown()

