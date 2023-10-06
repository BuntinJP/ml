import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

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

def transform_to_graph_data(X, y):
    # Convert dataframe to tensor
    x = torch.tensor(X.values, dtype=torch.float)
    
    # Create edges by connecting consecutive nodes (timestamps)
    edge_index = torch.tensor([list(range(X.shape[0]-1)), list(range(1, X.shape[0]))], dtype=torch.long)
    
    # Convert target to tensor
    y = torch.tensor(y.values, dtype=torch.float).view(-1, 1)
    
    # Create a PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 1st Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2nd Graph Convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Fully Connected Layer
        x = self.fc(x)
        return x

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
