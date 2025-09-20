import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from io_xyz import read_xyz
from featurize import build_graph
from models import GNNModel

# ================== Step 1. 加载数据 ==================
data_list = []
for file in glob.glob("data/*.xyz"):
    energy, coords = read_xyz(file)
    data_list.append(build_graph(coords, energy))

# 按 80/20 随机划分 (训练/测试)
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# 再从训练集中划分 10% 用作验证集
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16)
test_loader  = DataLoader(test_data, batch_size=16)

# ================== Step 2. 初始化模型 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(node_features=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_model_path = "results/models/gnn_model.pth"

# 用于记录loss曲线
train_losses = []
val_losses = []

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

    # 保存loss
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

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

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

mae  = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = math.sqrt(mse)
r2   = r2_score(y_true, y_pred)

# ================== Step 5. 保存指标到文件 ==================
metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2,
    "Final Train Loss": train_losses[-1],
    "Final Val Loss": val_losses[-1]
}

def save_metrics_to_json(metrics, filename="results/evaluation/metrics.json"):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

save_metrics_to_json(metrics)

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

save_energy_plot(y_true, y_pred)

# 额外新增：保存Loss曲线
def save_loss_curve(train_losses, val_losses, filename="results/figures/loss_curve.png"):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.savefig(filename)
    plt.close()

save_loss_curve(train_losses, val_losses)

# ================== Step 7. 打印测试结果 ==================
print(f"\n===== 测试集结果 =====")
print(f"MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
print(f"评估结果已保存到 results/evaluation/metrics.json 和 results/evaluation/metrics.csv")
print(f"能量对比图已保存到 results/figures/pred_vs_true.png")
print(f"📉 Loss 曲线已保存到 results/figures/loss_curve.png")
