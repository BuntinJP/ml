import os
import numpy as np
import pandas as pd
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset

cols = [3, 6, 9, 12, 15, 18, 19, 21, 22, 24, 25]

train_paths = [
    "20201127_1548_2_y_train.csv",
    "20201210_1112_2_y_train.csv",
    "20201210_1354_2_y_train.csv",
    "20201127_1840_5_y_train.csv",
    "20201130_1122_5_y_train.csv",
    "20201201_1429_5_y_train.csv",
    "20201203_1244_5_y_train.csv",
    "20201130_1808_6_y_train.csv",
    "20201203_1404_6_y_train.csv",
    "20201210_1610_6_y_train.csv",
    "20201127_1432_7_y_train.csv",
    "20201127_1701_7_y_train.csv",
    "20201203_1022_7_y_train.csv"
]
test_paths = [
    "20201126_1546_0_y_train.csv",
    "20201201_1230_0_y_train.csv",
    "20201201_1555_0_y_train.csv"
]

# デバイスの設定
DEVICE = torch.device("cpu")
# バッチサイズの指定（2^nで指定するのが慣例）
BATCHSIZE = 128
DIR = './dms_data/train/'

train_paths = map(lambda x: os.path.join(DIR, x), train_paths)
test_paths = map(lambda x: os.path.join(DIR, x), test_paths)
# EPOCHS数の指定。何回データ全体の学習を行うか（10を指定すると丸々データを10回訓練する）
EPOCHS = 10
# 1epochあたりの最大訓練＆テストデータサンプル数を指定（バッチサイズ*30とすると30個にデータが分割）
# 読み込んだデータがこの値より低いと関係ない
N_TRAIN_EXAMPLES = BATCHSIZE * 50
N_VALID_EXAMPLES = BATCHSIZE * 20
# 正規化数値
MIN_VAL = 1.0
MAX_VAL = 5.0


class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(LSTMModel, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])  # 最後の時間ステップの出力を取得
    return out


def normalize_target_variable(y_train, y_test, min_val, max_val):
  y_train_s = (y_train - min_val) / (max_val - min_val)
  y_test_s = (y_test - min_val) / (max_val - min_val)
  return y_train_s, y_test_s

# モデルの定義


def define_model(features_combi):
  # 層の数、各層のユニット数、およびドロップアウトの割合をOptunaから受け取り、MLPのモデルを構築
  # n_layersで層の数を指定
  n_layers = 3
  layers = []

  # 入力層の入力サイズを指定する。最初は特徴量の数に合わせる
  in_features = features_combi
  # 定義した層数分を回して、ユニット数＆
  for i in range(n_layers):
    # 出力数を指定する（次の層のユニット数になる（in_featuresになる））
    out_features = in_features + 2
    # nn.Linearは線形変換を行うためのモジュールで、全結合層を生成。入力と重み行列の行列積を計算し、バイアスを加えることで線形変換を行う。
    # バイアスをTrueにすることでバイアスを追加する（default=True)
    layers.append(nn.Linear(in_features, out_features, bias=True))
    # 活性化関数を指定。非線形に変換する。
    # 使える活性化関数はドキュメントへ→『Non-linear Activations (weighted sum, nonlinearity)』
    # https://pytorch.org/docs/stable/nn.html#non-linear-activations-other
    layers.append(nn.ReLU())
    # ドロップアウトは、訓練時にランダムに一部のノードを無効にすることで、過学習を防ぐ
    # ドロップアウトの確率を指定する(0.5だったら50%)
    p = 0.5
    layers.append(nn.Dropout(p))
    # 次の入力は前の出力層になるため数を同じに
    in_features = out_features
  # print(layers)
  # 最後に出力層として出力サイズを指定（回帰の場合は1に）
  layers.append(nn.Linear(in_features, 1))
  # print(layers)
  # print("-----")
  # 回帰問題ではLogSoftmaxは不要なので削除
  # layers.append(nn.LogSoftmax(dim=1))

  return nn.Sequential(*layers)


def normalize_target_variable(y_train, y_test, min_val, max_val):
  y_train_s = (y_train - min_val) / (max_val - min_val)
  y_test_s = (y_test - min_val) / (max_val - min_val)
  return y_train_s, y_test_s

# ローカルのCSVファイルからデータを読み込む関数


def load_local_data(csv_train_path, csv_test_path):
  # 選択する特徴量を決める
  # #目的変数を決める
  objective_variable = "oss"

  # 引数のcsv_train_path, csv_test_pathからデータをtrai_dataとtrain_combined_dataとtest_combined_dataに追加していく
  train_combined_data = []
  test_combined_data = []
  for train_path, test_path in zip(csv_train_path, csv_test_path):
    train_data = pd.read_csv(train_path, usecols=cols)
    test_data = pd.read_csv(test_path, usecols=cols)
    train_combined_data.append(train_data)
    test_combined_data.append(test_data)

  # リストに追加されていったデータを1つに繋げる
  df_train = pd.concat(train_combined_data, axis=0, ignore_index=True)
  df_test = pd.concat(test_combined_data, axis=0, ignore_index=True)

  # nanなどのデータを削除
  df_train = df_train.dropna()
  df_test = df_test.dropna()
  df_train = df_train.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)

  # 特徴量とラベルの分割
  # 訓練用の特徴量とラベル
  X_train = df_train[features].values
  y_train = df_train[objective_variable].values
  # テスト用の特徴量とラベル
  X_valid = df_test[features].values
  y_valid = df_test[objective_variable].values

  # ラベルの正規化
  y_train, y_valid = normalize_target_variable(
      y_train, y_valid, MIN_VAL, MAX_VAL
  )

  # PyTorchのテンソルに変換
  # PyTorchでのデータの扱いに適した形式にデータを変換 dtypeでデータ型を選択
  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
  y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
  # DataLoaderの作成
  # 同じインデックス位置のデータをラップ（まとめ）する
  '''tensor1 = torch.tensor([1, 2, 3])
       tensor2 = torch.tensor([4, 5, 6])
       dataset = TensorDataset(tensor1, tensor2)
       (tensor(1), tensor(4))
       (tensor(2), tensor(5))
       (tensor(3), tensor(6))'''
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

  '''DataLoaderを行うことで指定したバッチサイズにdata(特徴量)target(ラベル)に分けてくれる。
        (tensor(1), tensor(4))
        (tensor(2), tensor(5))
        (tensor(3), tensor(6))
        batch_size = 2
        Batch Data: tensor([1, 2])
        Batch Target: tensor([4, 5])
        Batch Data: tensor([3])
        Batch Target: tensor([6])
        for batch in data_loader:
          data, target = batch
        print("Batch Data:", data)
        print("Batch Target:", target)'''
  train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=False)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False)

  return train_loader, valid_loader

# RMSEを計算する関数


def calculate_rmse(predictions, targets):
  return torch.sqrt(nn.functional.mse_loss(predictions, targets))


def combi(n):
  global features_name
  comb_list = list(itertools.combinations(features_name, n))
  return comb_list


features_name = [
    "m_speed_stddev_480",
    "m_acceleration_stddev_480",
    "m_jerk_stddev_480",
    "m_steering_stddev_480",
    "AccelInput_stddev_480",
    "BrakeInput_stddev_480",
    "realtime steering entropy_1100",
    "realtime steering entropy_1100_stddev_480",
    "perclos",
]

# 目的関数
if __name__ == "__main__":
  train_combined_data = []
  test_combined_data = []
  for train_path in train_paths:
    train_data = pd.read_csv(train_path, usecols=cols)
    train_combined_data.append(train_data)

  for test_path in test_paths:
    test_data = pd.read_csv(test_path, usecols=cols)
    test_combined_data.append(test_data)
  train = pd.concat(train_combined_data, axis=0, ignore_index=True)
  test = pd.concat(test_combined_data, axis=0, ignore_index=True)

  # Pandasのデータフレームを作成
  df_train = pd.DataFrame(train)
  df_test = pd.DataFrame(test)

  # 欠損値の削除とインデックスの再設定
  df_train = df_train.dropna()
  df_test = df_test.dropna()
  df_train = df_train.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)

  result_list = []

  start = time.perf_counter()
  for iteration in range(1):
    print("iteration:", iteration)
    count = 0
    cnt = 0
    min_val = 1.0
    max_val = 5.0
    rmse_min = []
    feature_min = []

    for n in range(1, 10):
      x_col = combi(n)
      min_rmse = float("inf")
      f = -1
      print(len(x_col))
      for i in range(len(x_col)):
        features = x_col[i]
        X_train = df_train[list(features)].values
        y_train = df_train["oss"].values

        # テストデータの特徴量を選択
        X_test = df_test[list(features)].values
        y_test = df_test["oss"].values

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 正規化
        y_train_s, y_test_s = normalize_target_variable(
            y_train, y_test, min_val, max_val
        )

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_s, dtype=torch.float32)
        X_valid_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_test_s, dtype=torch.float32)
        # DataLoaderの作成
        # 同じインデックス位置のデータをラップ（まとめ）する

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

        train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False)
        # モデルの生成
        # Optunaのトライアル（ハイパーパラメータの探索）から受け取ったハイパーパラメータを使用して、モデルのアーキテクチャを構築。nn.Module を返す
        model = LSTMModel(input_size=n, hidden_size=64, num_layers=3, output_size=1).to(DEVICE)

        # オプティマイザの生成F
        optimizer_name = "Adam"  # 最適化アルゴリズムの選択
        lr = 0.001  # 学習率の選択
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=0.0001)  # PyTorchの最適化アルゴリズムの作成 weight_decay=0.0001は正則化L2項

        # モデルのトレーニング
        # EPOCHS数だけ回す（データ件数が10でBachSizeが2なら、5回繰り返すと、10件のデータを処理。 この1サイクルのことをEpochと呼ぶ。
        for epoch in range(EPOCHS):
          # model.train()で訓練モードに。訓練モードにすることでdropoutを有効にする
          # 訓練時にランダムに一部のユニットを無効にすることで、異なる部分ネットワークを学習させ、モデルが特定のパターンに依存しすぎないようにする。
          model.train()
          # batch_idxにはtrain_loaderの中身に入っているバッチ数が、dataには特徴量、testにはラベルが入る（データセット取得の時にload_local_dataでshuffleがFalseなら順番に）
          for batch_idx, (data, target) in enumerate(train_loader):
            # トレーニングデータを制限（現在の繰り返しでデータ数より上回るを処理を行うとループを抜ける（ほぼ使わない））
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
              break
            # .to()はPyTorchのテンソルやモデルを指定したデバイスに移動するためのメソッド。CPUかGPU(cuda)を指定できる
            data, target = data.to(DEVICE), target.to(DEVICE)
            # 逆伝播で各パラメータの勾配（偏微分）を計算し、それを使用してモデルのパラメータを更新するが、勾配が累積するためミニバッチごとに勾配を0にする。
            optimizer.zero_grad()
            # modelに特徴量を入れて学習させ予測値を出す
            output = model(data)
            # output(予測値)とラベルの誤差をRMSEで求める。（squeeze()はテンソルからサイズが1の次元を取り除く。outputの最後にはgrad_fn=<AddmmBackward0>のPytorchの自動微分が入ってるため。
            loss = calculate_rmse(output.squeeze(), target.float())

            # loss(損失)に対する各パラメータの勾配（偏微分）を計算する。（逆伝播）
            loss.backward()
            # 設定したPyTorchの最適化アルゴリズムが計算された勾配を使用して、モデルのパラメータを更新
            optimizer.step()

          # モデルの検証
          # 訓練モードで作ったモデルで検証を行う。そのため検証モードにし、Dropoutを無効にして訓練データを全て使用するようにする。
          model.eval()
          # 検証モードのため勾配の計算を無効にする。
          with torch.no_grad():
            # 1EPOCHS分をvalid_lossに格納する。（最後に平均を取るため）
            valid_loss = 0.0
            # 検証用のデータをdata,targetに分ける。
            for batch_idx, (data, target) in enumerate(valid_loader):
              # 検証データを制限
              if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                break
              # .to()はPyTorchのテンソルやモデルを指定したデバイスに移動するためのメソッド。CPUかGPU(cuda)を指定できる
              data, target = data.to(DEVICE), target.to(DEVICE)
              # 訓練で作成したモデルに検証用データを入れて予測値を出す
              output = model(data)
              # 予測値と正解ラベルの誤差を1EPOCHS分を格納する変数に追加
              valid_loss += calculate_rmse(output.squeeze(), target.float()).item()
          # RMSEの計算
          # valid_loaderのlengthでvalid_lossを割る。valid_loaderの長さはミニバッチの個数に当たるため。
          rmse = (valid_loss / len(valid_loader))

          if rmse < min_rmse:
            min_rmse = rmse
            f = i  # fは特徴量の組み合わせのインデックスを表してる。
      rmse_min.append(min_rmse)  # 各特徴量毎に最小のRMSEを格納
      feature_min.append(x_col[f])  # 各特徴量毎に最適な特徴量の組み合わせを格納
    print(rmse_min)
    result_list.append({
        'iteration': iteration,
        'feature_min': feature_min,
        'rmse_min': rmse_min,
        'rmse_mean': np.mean(rmse_min)
    })
    # 結果をDataFrameに変換してCSVに保存
  result_df = pd.DataFrame(result_list)
  result_df['rmse_min'] = result_df['rmse_min'].apply(lambda x: ', '.join(map(str, x)))
  result_df[['1', '2', '3', '4', '5', '6', '7', '8', '9']] = pd.DataFrame(result_df['rmse_min'].str.split(',').tolist(), dtype=float)
  result_df.drop(['rmse_min'], axis=1, inplace=True)
  result_df.to_csv('optuna/3/PytorchNN_oss_固定/results.csv', index=False)

  end = time.perf_counter() - start
  elapsed_minutes = int(end // 60)
  elapsed_seconds = int(end % 60)
  print("----------------------")
  print(result_df)
  print(f"経過時間: {elapsed_minutes}分 {elapsed_seconds}秒")
  print("----------------------")
