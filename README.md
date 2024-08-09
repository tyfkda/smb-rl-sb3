smb-rl-sb3
==========

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

### 学習例

<https://youtu.be/mlSjsejrrZY>

[![スーパーマリオのクッパ面を強化学習でクリア](http://img.youtube.com/vi/mlSjsejrrZY/0.jpg)](https://www.youtube.com/watch?v=mlSjsejrrZY)

#### 学習済みデータを動かすには

gitのブランチ`feature/world1-4`をチェックアウトして、

```sh
$ python main.py --movement complex --sb3_algo PPO \
  --color --skip-frame=2 --stage=4 \
  --replay=trained_model/model_world1-4.zip \
  --seed=12699629529116784663
```

### 参考

  * [PyTorchチュートリアル（日本語翻訳版）](https://yutaroogawa.github.io/pytorch_tutorials_jp/)
  * [Super Mario Bros. with Stable-Baseline3 PPO](https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo)

### ブログ記事

[スーパーマリオの強化学習を動かす（Stable Baselines 3）](https://tyfkda.github.io/blog/2024/08/07/smb-rl-sb3.html)
