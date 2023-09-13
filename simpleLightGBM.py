import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

DATA_DIR = './dms_data'

def load_and_preprocess_data(file_path):
    """CSV読み込んで前処理"""
    df = pd.read_csv(file_path)
    df = df.drop(['timestamp'], axis=1)
    return df.dropna()

def train_lightgbm(X, y):
    """LightGBMで学習させる関数。"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    eval_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, valid_sets=eval_data)
    return model

def get_data_from_directory(directory_path):
    """指定ディレクトリ内のCSV全部読んで結合"""
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    data_frames = [load_and_preprocess_data(os.path.join(directory_path, file)) for file in files]
    return pd.concat(data_frames, ignore_index=True)

def evaluate_model(model, X, y):
    """モデルの評価。RMSE"""
    predictions = model.predict(X)
    return np.sqrt(mean_squared_error(y, predictions))

# train読み込み
train_data = get_data_from_directory(os.path.join(DATA_DIR, 'train'))

X_oss = train_data.drop(['oss', 'Sleepiness'], axis=1)
y_oss = train_data['oss']

X_sleepiness = train_data.drop(['oss', 'Sleepiness'], axis=1)
y_sleepiness = train_data['Sleepiness']

# モデル学習開始
model_oss = train_lightgbm(X_oss, y_oss)
model_sleepiness = train_lightgbm(X_sleepiness, y_sleepiness)

# test読み込み
test_data = get_data_from_directory(os.path.join(DATA_DIR, 'test'))

X_test_oss = test_data.drop(['oss', 'Sleepiness'], axis=1)
y_test_oss = test_data['oss']

X_test_sleepiness = test_data.drop(['oss', 'Sleepiness'], axis=1)
y_test_sleepiness = test_data['Sleepiness']

# model評価
oss_rmse = evaluate_model(model_oss, X_test_oss, y_test_oss)
print(f"ossモデルのRMSE: {oss_rmse}")

sleepiness_rmse = evaluate_model(model_sleepiness, X_test_sleepiness, y_test_sleepiness)
print(f"眠気モデルのRMSE: {sleepiness_rmse}")
