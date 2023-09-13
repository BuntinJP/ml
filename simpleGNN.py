import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F

DATA_DIR = 'dms_data'

def load_and_preprocess_data(file_path):
    """CSVを読み込んでちょっと前処理する関数。"""
    df = pd.read_csv(file_path)
    df = df.drop(['timestamp'], axis=1)
    return df.dropna()

def get_data_from_directory(directory_path):
    """指定ディレクトリのCSV全部を読み込んで結合する関数。"""
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    data_frames = [load_and_preprocess_data(os.path.join(directory_path, file)) for file in files]
    return pd.concat(data_frames, ignore_index=True)

# トレインデータとテストデータを読み込む
train = get_data_from_directory(os.path.join(DATA_DIR, 'train'))
test = get_data_from_directory(os.path.join(DATA_DIR, 'test'))

# 特徴量を定義
features = [
    'm_speed', 'm_speed_var_480', 'm_speed_stddev_480', 'm_acceleration',
    'm_acceleration_var_480', 'm_acceleration_stddev_480', 'm_jerk',
    'm_jerk_var_480', 'm_jerk_stddev_480'
]
X_train, y_train, X_test, y_test = train[features], train['oss'], test[features], test['oss']

# ここからは元のGNNのコード
train_data = transform_to_graph_data(X_train, y_train)
test_data = transform_to_graph_data(X_test, y_test)

# y_trainをPyTorchのtensorに変換して形を整える
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)

# GNNモデルを定義
model = SimpleGNN(num_features=X_train.shape[1])

# GNNモデルを学習
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)

    loss = torch.nn.MSELoss()(out, y_train_tensor)
    loss.backward()
    optimizer.step()

# GNNモデルを評価
model.eval()
preds = model(test_data)
rmse = np.sqrt(mean_squared_error(y_test, preds.detach().numpy()))

print(rmse)
