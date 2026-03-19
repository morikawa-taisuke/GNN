import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# グラフのスタイル設定
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# データの読み込み
df = pd.read_csv('./final_graph_summary.csv')

# モデル名のマッピング（論文の表記に合わせるため）
domain_map = {
    'UGAT': 'WaveGAT',
    'UGCN': 'WaveGCN',
    'SpeqGAT': 'SpeqGAT',
    'SpeqGCN': 'SpeqGCN'
}

# --- 1. ディリクレエネルギーの比較グラフ (図2相当) ---
plt.figure(figsize=(8, 5))
de_data = df.groupby(['Domain', 'Range'])['Dirichlet_Energy_mean'].mean().reset_index()
de_data['Domain'] = de_data['Domain'].map(domain_map)

sns.barplot(data=de_data, x='Domain', y='Dirichlet_Energy_mean', hue='Range')
plt.title('Comparison of Dirichlet Energy (Mean)', fontsize=14)
plt.ylabel('Dirichlet Energy', fontsize=12)
plt.xlabel('Model Domain', fontsize=12)
plt.legend(title='Search Range')
plt.tight_layout()
plt.savefig('dirichlet_energy_comparison.png')
plt.close()

# --- 2. アテンション・エントロピーの層別推移グラフ (図4相当) ---
# GATモデルのみを抽出
gat_df = df[df['GNN'].str.contains('GAT')].copy()

# 各層のエントロピー列を一つの列にまとめる（Melt処理）
entropy_cols = [
    'attention_layer_0_entropy_mean',
    'attention_layer_1_entropy_mean',
    'attention_layer_2_entropy_mean'
]
gat_melted = gat_df.melt(
    id_vars=['GNN', 'Range'],
    value_vars=entropy_cols,
    var_name='Layer',
    value_name='Entropy'
)
# レイヤー名を数字（0, 1, 2）に変換
gat_melted['Layer'] = gat_melted['Layer'].str.extract('(\d+)').astype(int)
gat_melted['GNN'] = gat_melted['GNN'].map(domain_map)

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=gat_melted,
    x='Layer', y='Entropy',
    hue='GNN', style='Range',
    markers=True, dashes=False
)
plt.title('Attention Entropy Transition by Layer', fontsize=14)
plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Mean Entropy', fontsize=12)
plt.xticks([0, 1, 2])
plt.legend(title='Model & Range', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('attention_entropy_transition.png')
plt.close()

# --- 3. 音響環境別 (WaveType) の解析グラフ ---
plt.figure(figsize=(10, 5))
env_data = df[df['Range'] == 'all'].groupby(['WaveType', 'Domain'])['Dirichlet_Energy_mean'].mean().reset_index()
env_data['Domain'] = env_data['Domain'].map(domain_map)

sns.barplot(data=env_data, x='WaveType', y='Dirichlet_Energy_mean', hue='Domain')
plt.title('Dirichlet Energy by Environment (Range: all)', fontsize=14)
plt.ylabel('Dirichlet Energy', fontsize=12)
plt.xlabel('Acoustic Environment', fontsize=12)
plt.tight_layout()
plt.savefig('wavetype_analysis.png')
plt.close()

print("グラフの生成が完了しました: dirichlet_energy_comparison.png, attention_entropy_transition.png, wavetype_analysis.png")