import numpy as np
import torch
from io_xyz import read_xyz
from featurize import build_graph
from models import GNNModel

def perturb_coords(coords, sigma=0.02, k_atoms=3):
    """对部分原子加高斯扰动"""
    coords_new = coords.copy()
    idx = np.random.choice(coords.shape[0], size=k_atoms, replace=False)
    coords_new[idx] += np.random.normal(0, sigma, coords_new[idx].shape)
    return coords_new

# ==== 加载最低能结构 ====
energy0, coords0 = read_xyz("data/example_lowest.xyz")  # 你在analyze.py里可以保存最低能量结构为这个文件

# ==== 加载GNN模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(node_features=3).to(device)
model.load_state_dict(torch.load("results/models/gnn_model.pth"))
model.eval()

def predict_energy(coords):
    data = build_graph(coords, 0.0).to(device)  # energy随便填
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, torch.zeros(data.x.size(0), dtype=torch.long, device=device))
    return out.item()

# 基准能量
E0 = predict_energy(coords0)

# ==== 扰动实验 ====
sigmas = [0.005, 0.01, 0.02, 0.05]
results = {}
for sigma in sigmas:
    diffs = []
    for _ in range(100):  # 每个sigma扰动100次
        perturbed = perturb_coords(coords0, sigma=sigma, k_atoms=3)
        E_new = predict_energy(perturbed)
        diffs.append(abs(E_new - E0))
    results[sigma] = (np.mean(diffs), np.std(diffs))

print("扰动实验结果 (sigma, meanΔE, stdΔE):")
for sigma, (mean_diff, std_diff) in results.items():
    print(f"{sigma:.3f}: meanΔE={mean_diff:.4f}, stdΔE={std_diff:.4f}")

# ==== 定义稳定性变量 ====
# S1 = ΔE随sigma的斜率
sigmas_arr = np.array(list(results.keys()))
mean_diffs = np.array([v[0] for v in results.values()])
S1, _ = np.polyfit(sigmas_arr, mean_diffs, 1)  # 线性拟合斜率

# S2 = ΔE标准差 / sigma 平均 能量波动标准差归一化指标
S2 = np.mean([std/s for s, (_, std) in results.items()])

print(f"Stability variables: S1={S1:.4f}, S2={S2:.4f}")
