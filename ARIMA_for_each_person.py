# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from os import path
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# %%
features = [
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
TRAIN_DIR = 'dms_data/train/'
TEST_DIR = 'dms_data/test/'

train_csv = '20201126_1546_0_y_train.csv'
test_csv = train_csv.replace('train', 'test')

# %%


def solveOne(train_csv_path, test_csv_path, features, target, file_name=''):
  # train_df = pd.read_csv(path.join(TRAIN_DIR, train_csv))
  # test_df = pd.read_csv(path.join(TEST_DIR, test_csv))
  train_df = pd.read_csv(train_csv_path)
  test_df = pd.read_csv(test_csv_path)
  train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
  test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
  train_df.set_index('timestamp', inplace=True)
  test_df.set_index('timestamp', inplace=True)
  feature_scaler = MinMaxScaler()
  target_scaler = MinMaxScaler()
  train_df[features] = feature_scaler.fit_transform(train_df[features])
  test_df[features] = feature_scaler.transform(test_df[features])
  train_df[target] = target_scaler.fit_transform(train_df[[target]])
  test_df[target] = target_scaler.transform(test_df[[target]])
  model = ARIMA(train_df[target], order=(5, 1, 0))
  model_fit = model.fit()
  predictions = model_fit.forecast(steps=len(test_df))
  rmse = np.sqrt(mean_squared_error(test_df[target], predictions))
  print('Test RMSE: %.3f' % rmse)
  plt.title(f'{target} over Time')
  plt.xlabel('Timestamp')
  plt.ylabel(target)
  plt.plot(test_df.index, test_df[target], label='Actual')
  plt.plot(test_df.index, predictions, label='Predicted', color='red')
  plt.legend()
  if file_name != '':
    plt.savefig(path.join('./figure', file_name))
    # 画像を表示
    plt.show()
  return rmse

# %%


""" 
target => 予測対象のカラム
"""
# target = 'oss'
target = 'Sleepiness'

csvs = os.listdir(TRAIN_DIR)
for csv in csvs:
  if csv.endswith('y_train.csv'):
    train_path = path.join(TRAIN_DIR, csv)
    test_path = path.join(TEST_DIR, csv.replace('train', 'test'))
    # test_path = train_path
    png_file_name = csv.replace('y_train.csv', f'arima-{target}.png')
    print(png_file_name.replace('y_train.csv', ''))
    solveOne(train_path, test_path, features, target, png_file_name)
    print('------------------------')
