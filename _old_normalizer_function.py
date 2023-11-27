def normalize_data(series):
    scaler = MinMaxScaler()
    series = series.values.reshape(-1, 1) 
    series = scaler.fit_transform(series)
    series = series.ravel()
    return pd.Series(series)

# 正規化
y_oss = normalize_data(y_oss)
y_sleepiness = normalize_data(y_sleepiness)
