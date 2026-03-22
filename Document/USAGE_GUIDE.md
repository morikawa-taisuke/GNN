# 実行手順書 (Usage Guide)

本プロジェクト（GNN Base Audio Enhancement）における、データ準備から学習、そして評価までの一般的な実行フローを解説します。

## 1. データの準備
音声データとノイズ・残響データを合成し、機械学習モデルに入力できる形式に前処理します。

### ノイズの合成 (`make_mixdown.py`)
クリーンな音声に対し、任意のSNR（Signal-to-Noise Ratio）でノイズを付加した音声（ミックスダウン）を作成します。
```bash
python scripts/make_mixdown.py
```
*※ スクリプト末尾の `if __name__ == "__main__":` 以下のパス（`speech_dir`, `noise_dir`）を環境に合わせて書き換えてから直接実行してください。*

### データセット形式への変換 (`make_dataset.py`)
モデルの学習・推論を高速化するため、入力と正解信号をペアにしたデータ（`.npz`形式など）を生成する各種ユーティリティ関数が含まれています。

---

## 2. モデルの学習と推論
システムのエントリポイントは `experiments/` ディレクトリの下に配置されています。
本系リポジトリはコマンドライン引数（argparse等）を使用せず、**各スクリプトの末尾にある `if __name__ == "__main__":` ブロックのハードコードされたパラメータを書き換えて実行する運用**となっています。

### 周波数領域モデル（SpeqGNN等）の実行
STFTを用いてスペクトログラム領域でGNNを用いたモデルを学習します。
```bash
python experiments/main_Speq.py
```
**主な変更パラメータ（ファイル末尾）:**
- `model_list`: "GCN", "GAT", "UNet", "ConvTasNet" 等から使用モデルを選択します
- `wave_types`: "noise_only", "reverb_only", "noise_reverb" 等
- `stft_params`: `n_fft` や `hop_length` の設定
- `num_node`: グラフのノード数（デフォルト 32）

### 時間領域モデル（WaveUGNN等）の実行
波形データをそのまま1次元配列としてGNNに入力するモデルの学習に用います。
```bash
python experiments/main_Wave.py
```

---

## 3. 推論結果の評価
学習によって出力された分離・強調済み音声（WAVファイル）の客観的品質（PESQ, STOI, SISDR等）を評価します。

```bash
python experiments/evaluation/All_evaluation.py
```
特定のCSVデータセットに対する評価結果を一覧でまとめる場合は、同ディレクトリ内の `CSV_eval.py` 等も利用できます。
