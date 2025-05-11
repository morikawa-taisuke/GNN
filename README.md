# RelNet (Graph Neural Network) Implementation

このプロジェクトは、PyTorchを使用したRelNet（グラフニューラルネットワーク）の実装です。

## 必要条件

- Python 3.7以上
- PyTorch 2.0.0以上
- PyTorch Geometric 2.3.0以上
- NumPy 1.21.0以上
- Matplotlib 3.4.0以上

## インストール方法

1. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

2. PyTorch Geometricの追加インストールが必要な場合:
```bash
pip install torch-geometric
```

## 使用方法

プログラムを実行するには、以下のコマンドを実行してください：

```bash
python relnet.py
```

## 実装の詳細

- `RelNet`クラスは3層のGCN（Graph Convolutional Network）を実装しています
- 入力次元、隠れ層の次元、出力次元を指定可能です
- ドロップアウトとReLU活性化関数を使用しています
- 学習にはAdamオプティマイザーを使用しています

## サンプルデータ

現在の実装では、ランダムに生成されたグラフデータを使用しています：
- 100ノード
- 16次元の特徴量
- 3つのクラス
- 200のエッジ 