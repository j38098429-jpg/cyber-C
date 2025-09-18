import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from io_xyz import read_xyz
from featurize import build_graph
from models import GNNModel

# ==== 加载数据 ====
data_list = []
energies = []
coords_list = []
for file in glob.glob("data/*.xyz"):
    energy, coords = read_xyz(file)
    data_list.append(build_graph(coords, energy))
    energies.append(energy)
    coords_list.append(coords)

energies = np.array(energies)
coords_list = np.array(coords_list)

# ==== 统计指标 ====
mean_energy = energies.mean()
var_energy = energies.var()
skew_energy = ((energies - mean_energy)**3).mean() / (energies.std()**3)
print(f"Mean={mean_energy:.3f}, Var={var_energy:.3f}, Skew={skew_energy:.3f}")

# ==== 能量分布图 ====
def save_energy_distribution(energies, filename="results/figures/energy_distribution.png"):
    plt.hist(energies, bins=50, color="blue", alpha=0.7)
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Energy Distribution")
    plt.savefig(filename)
    plt.close()

save_energy_distribution(energies)

# ==== 找最低能量结构 ====
min_idx = np.argmin(energies)
print(f"Lowest energy = {energies[min_idx]:.3f}, at structure #{min_idx}")

# ==== 用GNN预测（验证模型效果） ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(node_features=3).to(device)
model.load_state_dict(torch.load("results/models/gnn_model.pth"))  # 从train.py保存的模型
model.eval()

loader = DataLoader(data_list, batch_size=32)
preds, trues = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(out.squeeze().cpu().numpy())
        trues.append(batch.y.cpu().numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

# ==== 对比真实能量和预测能量 ====
def save_energy_plot(y_true, y_pred, filename="results/figures/pred_vs_true.png"):
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.title("GNN Energy Prediction (Task2)")
    plt.savefig(filename)
    plt.close()

save_energy_plot(trues, preds)

# 输出评估结果
print(f"\n预测 vs 真实能量的图已保存到 results/figures/pred_vs_true.png")
