
### 要件

  * Python 3.12＋仮想環境
  * Cコンパイラ (nes-py用)

#### GLU

pygletがGLUを用いているため、別途インストールしておく必要がある。

Windows/WSL2の場合：[Install OpenGL on Ubuntu in WSL](https://gist.github.com/Mluckydwyer/8df7782b1a6a040e5d01305222149f3c)

```sh
$ apt install mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev
```


### 初期設定

サブモジュールの取得：

```sh
$ git submodule update --init --recursive
```

Python仮想環境を用意・有効にした上で、

```sh
$ make setup
```


### 動作テスト

```sh
$ python play_randomly.py
```


### トレーニング

```sh
$ python main.py --movement simple --sb3_algo PPO
```

### 再生

```sh
$ python main.py --movement simple --sb3_algo PPO --replay
```


### 参考

  * [PyTorchチュートリアル（日本語翻訳版）](https://yutaroogawa.github.io/pytorch_tutorials_jp/)
  * [Super Mario Bros. with Stable-Baseline3 PPO](https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo)
