# GNN Base Audio Enhancement

本リポジトリは、グラフニューラルネットワーク（GNN）とU-Netを組み合わせた**音源強調およびノイズ除去モデルの開発・評価用リポジトリ**です。
時間領域波形や周波数スペクトログラムを入力とし、空間・チャネル間の依存関係をGNNで捉えることで、高度な音響強調・分離（UGNN, SpeqGNN, WaveUGNN等）を実現します。

## 🤖 RAG（AIアシスタント）を利用する際の手引き
このリポジトリは、AI（LLM）に対してコードの仕様や動作を問い合わせ（RAG）しやすいように最適化されています。
AIに質問する際は、「`src/models/WaveUGNN.py` のボトルネック部分の構造を教えて」や、「`experiments/main_Wave.py` の学習ループの流れを説明して」のように、**具体的なファイル名**を指定すると、より正確な回答が得られます。各主要モジュールには詳細なDocstringが記載されています。

## 📂 リポジトリの構成

- **`src/`** : プロジェクトのコアアーキテクチャ
  - `dataset/` : データローダーやDatasetクラス群（`CsvDataset.py`等）
  - `models/` : モデルアーキテクチャ定義（`SpeqGNN.py`, `WaveUGNN.py`, `ConvTasNet_models.py` 等）
  - `mymodule/` : カスタム損失関数（LossFunction）や共通関数群
- **`scripts/`** : データセット作成や一時的な分析ツール
  - 音声長の確認 (`check_audio_durations.py`) やデータサンプリング (`random_choice.py`) などのユーティリティ
  - データセットの合成・生成用スクリプト（`make_mixdown.py`, `make_dataset.py`等）
- **`experiments/`** : 学習・推論実行用エントリポイント
  - `main_Speq.py` : **周波数領域（スペクトログラム）** の学習・推論。
  - `main_Wave.py` : **時間領域（波形）** の学習・推論。
  - `main_overlap_add.py` : 長尺音声に対するオーバーラップ・アド法を用いた推論用。
  - `evaluation/` : SISDR, PESQ, STOIなどの客観評価用スクリプト群。
- **`analysis/` & `check_node/`** : グラフ構造の詳細分析や、生成音声の品質分析ツール。
- **`Document/`** : 詳細な設計仕様やアルゴリズム構想に関するドキュメント群。

## ⚙️ 動作環境・インストール

必要なパッケージをインストールして環境を構築します。
```bash
pip install -r requirements.txt
```
※ PyTorch Geometric等については、PyTorchとCUDAのバージョンに合わせて適切なホイールを追加インストールしてください。

## 🚀 使用方法・実行手順

### 1. データの準備
`scripts/` 配下のスクリプトを使用して学習用データを作成します。音声長のチェックなどが必要な場合は適宜ユーティリティを使用してください。

### 2. 学習の実行
ルートディレクトリから、`experiments/` 内のスクリプトを実行します。モジュールパスを正しく解決するため、必ずルートディレクトリから実行してください。
```bash
python experiments/main_Speq.py
# または
python experiments/main_Wave.py
```

### 3. 評価の実行
客観評価やプロファイリングは `experiments/evaluation/` 配下のスクリプトを使用します。