# 青木拓海 tanicsrg-hi

- [青木拓海機械学習ホーム](https://scrapbox.io/tanicsrg-hi/%E9%9D%92%E6%9C%A8%E6%8B%93%E6%B5%B7)
- [py-feat](https://scrapbox.io/tanicsrg-hi/青木拓海_py-feat)
- [py-feat_result_sample](https://scrapbox.io/tanicsrg-hi/G2024_青木拓海_py-feat_結果サンプル用)
  [まとめ時使用](https://scrapbox.io/tanicsrg-hi/青木_時系列解析_ARIMA_その他スライド用)

---

# 引き継ぎ内容

- [機械学習帳](https://chokkan.github.io/mlnote/index.html)

- 使用した環境
  - OS: RHEL 9.3
  - Python: 3.9

[ゼミ Google ドライブ](https://drive.google.com/drive/folders/1M1RS79CTCbjAgRT_AS1QDQztMJlX-bTp?usp=drive_link)

**内容**

- パッケージ(`$ pip freeze > requirements.txt`)まとめ
- ml_hikitugi 引き継ぎコード

# !!!注意!!!

pip のグローバルではなく、virtualenv(venv)を利用して環境を分けたほうが良い。
dlib や py-feat,sktime など、あまり用いられていないパッケージは、OS によっては動作が安定しないことがある。
しかしどうしても使わないといけない、使いたい時は、ビルドする必要がある。
そのビルドで失敗し、またやり直す、などは結構やったが、依存関係が壊れる可能性がある。
依存関係が壊れるとクソだるい。しかし、venv なら壊れても環境を消すだけで大丈夫。
(numpy,pandas,tensorflow,lightGBM などは大丈夫)

```
(jupyter-env) [buntin@buntin.tech ~]$ python -V
Python 3.9.18
(jupyter-env) [buntin@buntin.tech ~]$ pip list
Package                      Version
---------------------------- ------------
absl-py                      1.4.0
aiofiles                     22.1.0
aiosqlite                    0.18.0
alembic                      1.13.0
anyio                        3.6.2
argon2-cffi                  21.3.0
argon2-cffi-bindings         21.2.0
arrow                        1.2.3
asttokens                    2.2.1
astunparse                   1.6.3
async-generator              1.10
attrs                        22.2.0
autograd                     1.5
av                           11.0.0
Babel                        2.12.1
backcall                     0.2.0
beautifulsoup4               4.12.0
bleach                       6.0.0
cachetools                   5.3.0
celluloid                    0.2.0
certifi                      2022.12.7
cffi                         1.15.1
charset-normalizer           3.1.0
cmake                        3.26.4
cmdstanpy                    1.2.0
colorama                     0.4.6
colorlog                     6.8.0
comm                         0.1.3
contourpy                    1.0.7
convertdate                  2.4.0
cycler                       0.11.0
Cython                       3.0.3
debugpy                      1.6.6
decorator                    5.1.1
defusedxml                   0.7.1
dlib                         19.24.2
easing-functions             1.0.4
ephem                        4.1.4
exceptiongroup               1.1.1
executing                    1.2.0
fastjsonschema               2.16.3
filelock                     3.12.0
flatbuffers                  23.5.8
fonttools                    4.39.3
fqdn                         1.5.1
future                       0.18.3
gast                         0.4.0
gitdb                        4.0.10
GitPython                    3.1.31
google-auth                  2.18.0
google-auth-oauthlib         1.0.0
google-pasta                 0.2.0
greenlet                     3.0.2
grpcio                       1.54.0
h11                          0.14.0
h5py                         3.8.0
holidays                     0.34
idna                         3.4
imageio                      2.33.0
importlib-metadata           6.8.0
importlib-resources          6.0.1
ipykernel                    6.22.0
ipython                      8.12.0
ipython-genutils             0.2.0
ipywidgets                   8.0.6
isoduration                  20.11.0
japanize-matplotlib          1.1.3
jax                          0.4.12
jaxlib                       0.4.12
jedi                         0.18.2
Jinja2                       3.1.2
joblib                       1.2.0
json5                        0.9.11
jsonpointer                  2.3
jsonschema                   4.17.3
jupyter                      1.0.0
jupyter_client               8.1.0
jupyter-console              6.6.3
jupyter_core                 5.3.0
jupyter-events               0.6.3
jupyter_server               2.5.0
jupyter_server_fileid        0.8.0
jupyter-server-mathjax       0.2.6
jupyter_server_terminals     0.4.4
jupyter_server_ydoc          0.8.0
jupyter-ydoc                 0.2.3
jupyterlab                   3.6.3
jupyterlab-fonts             2.1.1
jupyterlab-git               0.41.0
jupyterlab-pygments          0.2.2
jupyterlab_server            2.22.0
jupyterlab-widgets           3.0.7
keras                        2.12.0
kiwisolver                   1.4.4
kornia                       0.7.0
lazy_loader                  0.3
libclang                     16.0.0
lightgbm                     3.3.5
lit                          16.0.5.post0
LunarCalendar                0.0.9
lxml                         4.9.3
Mako                         1.3.0
Markdown                     3.4.3
MarkupSafe                   2.1.2
matplotlib                   3.7.1
matplotlib-inline            0.1.6
mistune                      2.0.5
ml-dtypes                    0.1.0
mpmath                       1.3.0
nbclassic                    0.5.4
nbclient                     0.7.2
nbconvert                    7.2.10
nbdime                       3.1.1
nbformat                     5.8.0
nest-asyncio                 1.5.6
networkx                     3.1
nibabel                      5.1.0
nilearn                      0.10.2
nltools                      0.5.0
nodejs                       0.1.1
notebook                     6.5.3
notebook_shim                0.2.2
numexpr                      2.8.4
numpy                        1.23.5
nvidia-cublas-cu11           11.10.3.66
nvidia-cuda-cupti-cu11       11.7.101
nvidia-cuda-nvrtc-cu11       11.7.99
nvidia-cuda-runtime-cu11     11.7.99
nvidia-cudnn-cu11            8.5.0.96
nvidia-cufft-cu11            10.9.0.58
nvidia-curand-cu11           10.2.10.91
nvidia-cusolver-cu11         11.4.0.1
nvidia-cusparse-cu11         11.7.4.91
nvidia-nccl-cu11             2.14.3
nvidia-nvtx-cu11             11.7.91
oauthlib                     3.2.2
opencv-python                4.8.1.78
opt-einsum                   3.3.0
optional-django              0.1.0
optuna                       3.4.0
outcome                      1.2.0
packaging                    23.0
pandas                       2.0.0
pandocfilters                1.5.0
parso                        0.8.3
patsy                        0.5.3
pexpect                      4.8.0
pickleshare                  0.7.5
Pillow                       9.5.0
pip                          23.3.1
platformdirs                 3.2.0
pmdarima                     2.0.3
prometheus-client            0.16.0
prompt-toolkit               3.0.38
prophet                      1.1.4
protobuf                     4.23.0
psutil                       5.9.4
ptyprocess                   0.7.0
pure-eval                    0.2.2
py-feat                      0.6.1
pyasn1                       0.5.0
pyasn1-modules               0.3.0
pycparser                    2.21
Pygments                     2.14.0
PyMeeus                      0.5.12
pynv                         0.3
pyparsing                    3.0.9
pyproject-toml               0.0.10
pyrsistent                   0.19.3
PySocks                      1.7.1
python-dateutil              2.8.2
python-json-logger           2.0.7
pytz                         2023.3
PyWavelets                   1.5.0
PyYAML                       6.0
pyzmq                        25.0.2
qtconsole                    5.4.2
QtPy                         2.3.1
requests                     2.28.2
requests-oauthlib            1.3.1
rfc3339-validator            0.1.4
rfc3986-validator            0.1.1
rsa                          4.9
scikit-base                  0.5.2
scikit-image                 0.22.0
scikit-learn                 1.2.2
scipy                        1.10.1
seaborn                      0.12.2
selenium                     4.8.3
Send2Trash                   1.8.0
setuptools                   53.0.0
six                          1.16.0
sktime                       0.23.0
smmap                        5.0.0
sniffio                      1.3.0
sortedcontainers             2.4.0
soupsieve                    2.4
SQLAlchemy                   2.0.23
stack-data                   0.6.2
stanio                       0.3.0
statsmodels                  0.14.0
sympy                        1.12
tensorboard                  2.12.3
tensorboard-data-server      0.7.0
tensorflow                   2.12.0
tensorflow-estimator         2.12.0
tensorflow-io-gcs-filesystem 0.32.0
termcolor                    2.3.0
terminado                    0.17.1
threadpoolctl                3.1.0
tifffile                     2023.9.26
tinycss2                     1.2.1
toml                         0.10.2
tomli                        2.0.1
torch                        2.0.1
torch-geometric              2.3.1
torchaudio                   2.0.2
torchvision                  0.15.2
tornado                      6.2
tqdm                         4.66.1
traitlets                    5.9.0
trio                         0.22.0
trio-websocket               0.10.2
triton                       2.0.0
typing_extensions            4.5.0
tzdata                       2023.3
uri-template                 1.2.0
urllib3                      1.26.15
wcwidth                      0.2.6
webcolors                    1.13
webencodings                 0.5.1
websocket-client             1.5.1
Werkzeug                     2.3.4
wheel                        0.41.2
widgetsnbextension           4.0.7
wrapt                        1.14.1
wsproto                      1.2.0
xgboost                      2.0.2
y-py                         0.5.9
ypy-websocket                0.8.2
zipp                         3.16.2

[notice] A new release of pip is available: 23.3.1 -> 23.3.2
[notice] To update, run: pip install --upgrade pip
(jupyter-env) [buntin@buntin.tech ~]$
```
