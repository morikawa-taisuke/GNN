import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torchinfo import summary

class RelNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RelNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

def print_model_summary(model, input_dim, num_nodes, num_edges):
    # サンプル入力データを作成
    x = torch.randn((num_nodes, input_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # モデルのサマリーを表示
    print("\nRelNet Model Summary:")
    summary(model, input_data=(x, edge_index))

def main():
    # サンプルデータの作成
    num_nodes = 100
    num_features = 16
    num_classes = 3
    num_edges = 200
    
    # ランダムな特徴量とエッジを作成
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # トレーニングとテスト用のマスクを作成
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:80] = True
    test_mask[80:] = True
    
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    
    # モデルの初期化
    model = RelNet(input_dim=num_features, hidden_dim=32, output_dim=num_classes)
    
    # モデルのサマリーを表示
    print_model_summary(model, num_features, num_nodes, num_edges)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    # 学習
    for epoch in range(100):
        loss = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            acc = test(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

if __name__ == '__main__':
    main() 