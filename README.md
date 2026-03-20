# GNN Base Audio Enhancement

本リポジトリは、グラフニューラルネットワーク（GNN）とU-Netを組み合わせた**音源強調およびノイズ除去モデルの開発・評価用リポジトリ**です。
時間領域波形や周波数スペクトログラムを入力とし、残響特性やインパルス応答の特徴量を加味して音響信号をクリアにする高度なモデル群（UGNN, SpeqGNN 等）が含まれています。

## リポジトリの構成

後輩への引き継ぎを目的としてディレクトリ構成を整理し、役割ごとに分割しています。

- `src/` : プロジェクトのコアモジュール
  - `dataset/` : データローダーやDatasetクラス群（`CsvDataset.py`等）
  - `models/` : モデルアーキテクチャ定義（`SpeqGNN`, `WaveUGNN`, `ConvTasNet` 等）
  - `mymodule/` : カスタム損失関数や共通関数群
- `scripts/` : データ準備とツール
  - データセットの合成・生成用スクリプト（`make_mixdown.py`, `make_dataset.py`等）
  - JSONリスト作成・変換スクリプト群
- `experiments/` : 目的別の学習・推論実行用エントリポイント
  - `main_Speq.py` : **周波数領域（スペクトログラム）** のモデル用。STFTを内部で適用し、`SpeqGNN`等で学習・推論を行います。
  - `main_Wave.py` : **時間領域（波形）** のモデル用。1次元波形をそのまま入力とし、`Wave_UGNN`等で学習・推論を行います。
  - `main_overlap_add.py` : オーバーラップアド法を用いた長尺音声に対する推論スクリプト。
  - `evaluation/` : SISDR, PESQ, STOIなどの各種客観評価用スクリプト群（`All_evaluation.py` 等）。
- `analysis/` & `check_node/` : グラフ構造の詳細分析や、生成音声の品質分析ツール群。
- `Document/` : 詳細な設計仕様やアルゴリズム構想（[UGNN.md](Document/UGNN.md)等を格納）。

## 動作環境・インストール

1. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

2. （必要があれば）PyTorch Geometricの追加インストールなど、環境に応じた設定を行ってください。

## 使用方法・実行手順

### 1. データの準備
`scripts/data_prep/` 配下のスクリプトを使用して学習用データを作成します。
```bash
python scripts/data_prep/make_mixdown.py
```

### 2. 学習の実行
ルートディレクトリから、`experiments/` 内のスクリプトを実行します。実行時はモジュールパスを解決するため、ルートから実行することを推奨します。
```bash
python experiments/main_Speq.py
# または
python experiments/main_Wave.py
```

### 3. 評価の実行
客観評価やプロファイリングは `experiments/evaluate/` 配下のスクリプトを使用します。
```bash
python experiments/evaluate/All_evaluation.py
```