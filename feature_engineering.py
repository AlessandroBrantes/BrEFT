from ta import add_all_ta_features

def create_features(data, open_col, high_col, low_col, close_col, volume_col):
    data = add_all_ta_features(data, open=open_col, high=high_col, low=low_col, close=close_col, volume=volume_col)
    return data


