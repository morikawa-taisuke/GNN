# データローダー要件定義書

## 1. 概要

本ドキュメントは、音声強調モデルの学習・評価に必要なデータローダーの要件を定義します。このデータローダーは、
`EnhancementTrainer`クラスと連携して動作することを想定しています。

## 2. 基本要件

### 2.1. データセットクラス

#### 2.1.1. 学習・検証用データセット (`AudioDataset`)

- **入力**:
    - クリーン音声のディレクトリパス
    - ノイズ付き音声のディレクトリパス
    - サンプリングレート
    - 最大音声長（sec）

- **出力**:
    - ノイズ付き音声波形（torch.Tensor）
    - クリーン音声波形（torch.Tensor）
    - 各音声ファイルのパス

#### 2.1.2. テスト用データセット (`AudioDatasetTest`)

- **入力**:
    - ノイズ付き音声のディレクトリパス
    - サンプリングレート
    - 最大音声長（sec）

- **出力**:
    - ノイズ付き音声波形（torch.Tensor）
    - 音声ファイルのパス

### 2.2. データローダー設定

- バッチサイズ: モデルとGPUメモリに応じて設定可能
- シャッフル: 学習時はTrue、評価時はFalse
- pin_memory: GPU使用時はTrue推奨

## 3. 機能要件

### 3.1. 音声データの前処理

1. **音声の読み込み**
    - サポートするフォーマット: WAV
    - Tensor型でロード

2. **音声長の調整**
    - 指定された最大長でのパディングまたはトリミング
    - バッチ内の音声長を統一

### 3.2. データ拡張（オプション）

1. **音量調整**
    - ランダムなゲイン調整（±6dB範囲）

2. **時間シフト**
    - ランダムな時間シフト（最大長を超えない範囲）

### 3.3. エラー処理

1. **データ検証**
    - ファイルの存在確認
    - 音声フォーマットの検証

2. **エラーハンドリング**
    - 破損ファイルのスキップ
    - 適切なエラーメッセージの提供 (日本語)

## 4. 実装詳細

### 4.1. クラスインターフェース

```python
class AudioDataset(torch.utils.data.Dataset): 
    def **init**(self, clean_dir: str, noisy_dir: str, sample_rate: int, max_length: int, transform: Optional[Callable] = None): pass
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pass
```

### 4.2. 設定パラメータ
```python
dataset_config = { 'sample_rate': 16000, 
                   'max_length': 4, # 4sec
                   'transform': None }
dataloader_config = { 'batch_size': 16, 'shuffle': True, 'pin_memory': True }
```
## 5. 性能要件

- データ読み込みのレイテンシ: バッチあたり100ms以下
- メモリ使用量: 利用可能なRAMの50%以下
- GPUメモリ効率: 最適なバッチサイズでのメモリ使用

## 6. テスト要件

1. **単体テスト**
    - データ読み込みの正確性
    - 前処理の正確性
    - エラーハンドリングの動作確認

2. **統合テスト**
    - Trainerクラスとの連携確認
    - バッチ処理の整合性
    - メモリリークのチェック

