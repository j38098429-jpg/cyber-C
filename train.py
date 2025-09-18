import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
import math

from io_xyz import read_xyz
from featurize import build_graph
from models import GNNModel

# ================== Step 1. 加载数据 ==================
data_list = []
for file in glob.glob("data/*.xyz"):
    energy, coords = read_xyz(file)
    data_list.append(build_graph(coords, energy))

# 划分训练/验证/测试 (800/100/100)
train_data = data_list[:800]
val_data   = data_list[800:900]
test_data  = data_list[900:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

# ================== Step 2. 初始化模型 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(node_features=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_model_path = "results/models/gnn_model.pth"

# ================== Step 3. 训练循环 ==================
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # ---- 验证 ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out.squeeze(), batch.y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ---- 保存最佳模型 ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存新最佳模型到 {best_model_path}")

# ================== Step 4. 测试集评估 ==================
# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(out.squeeze().cpu().numpy())

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

mae  = mean_absolute_error(y_true, y_pred)
# 先计算MSE，然后手动计算平方根获取RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = math.sqrt(mse)
r2   = r2_score(y_true, y_pred)

# ================== Step 5. 保存指标到文件 ==================
# 保存测试指标到 JSON 文件
metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2,
    "Train Loss": avg_train_loss,
    "Val Loss": avg_val_loss
}

def save_metrics_to_json(metrics, filename="results/evaluation/metrics.json"):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

save_metrics_to_json(metrics)

# 保存测试指标到 CSV 文件
def save_metrics_to_csv(metrics, filename="results/evaluation/metrics.csv"):
    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(filename, index=False)

save_metrics_to_csv(metrics)

# ================== Step 6. 保存图表 ==================
def save_energy_plot(y_true, y_pred, filename="results/figures/pred_vs_true.png"):
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.title("GNN Energy Prediction")
    plt.savefig(filename)
    plt.close()

# 保存预测 vs 真实能量的图
save_energy_plot(y_true, y_pred)

# 输出测试结果
print(f"\n===== 测试集结果 =====")
print(f"MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
print(f"评估结果已保存到 results/evaluation/metrics.json 和 results/evaluation/metrics.csv")
print(f"能量对比图已保存到 results/figures/pred_vs_true.png")
