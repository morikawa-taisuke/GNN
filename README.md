# GNN Base Audio Enhancement

本リポジトリは、グラフニューラルネットワーク（GNN）とU-Netを組み合わせた**音源強調およびノイズ除去モデルの開発・評価用リポジトリ**です。
時間領域波形や周波数スペクトログラムを入力とし、残響特性やインパルス応答の特徴量を加味して音響信号をクリアにする高度なモデル群（UGNN, SpeqGNN 等）が含まれています。

## リポジトリの構成

後輩への引き継ぎを目的として、プロジェクトのディレクトリ構成を整理しました。

- `src/` : プロジェクトのコアモジュール
  - `dataset/` : データローダーやDatasetクラス群（CSV管理ベース）
  - `models/` : モデルアーキテクチャ定義（SpeqGNN, WaveUGNN, GraphEncoder等）
  - `utils/` : カスタム損失関数やGPU確認スクリプトなどの共通関数群
- `scripts/` : データ準備とツール
  - `data_prep/` : データセットの合成・生成用スクリプト（`make_mixdown.py`等）
  - `tools/` : JSONリスト作成・変換等のユーティリティ
- `experiments/` : 実験用エントリポイント
  - 訓練用のメインスクリプト群（旧 `main*.py` 群）
  - `evaluate/` : 各種評価用スクリプト（PESQ, STOI, SISDR、各種客観評価、ミュージカルノイズ解析等）
- `tests/` : 旧確認用スクリプトや検証用コード
- `Document/` : 詳細な設計仕様やアルゴリズム構想（モデルの詳細は[UGNN.md](Document/UGNN.md)などを参照）

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