import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('./tutorial.mplstyle')

# ================= 参数设置 =================
N = 50
alpha = 0.25
lam = 1.25
tau = 15.0
dt = 0.25
T = 200
steps = int(T / dt)
time = np.arange(0, T, dt)

# ================= 记忆模式定义 (优化重叠度) =================
P = 5
patterns = np.zeros((P, N))

# 模式 0: 17 到 30 (目标模式 B)
patterns[0, 17:31] = 1
# 模式 1: 每隔 4 个 (目标模式 C)
patterns[1, ::4] = 1
# 模式 2: 32 到 45 (低重叠背景)
patterns[2, 32:46] = 1
# 模式 3: 偏移 1 的条纹 (低重叠背景)
patterns[3, 1::4] = 1
# 模式 4: 偏移 2 的条纹 (低重叠背景)
patterns[4, 2::4] = 1

# ================= 计算权重矩阵 =================
W = np.zeros((N, N))
for mu in range(P):
    diff = patterns[mu] - alpha
    W += np.outer(diff, diff)
W = (lam / (N * alpha * (1 - alpha))) * W
np.fill_diagonal(W, 0)

# ================= 激活函数 =================
def F(I):
    val = 150.0 * np.tanh((I + 20.0) / 150.0)
    return np.maximum(0, val)

# ================= 模拟函数 (优化动力学) =================
def simulate(target_pattern_idx):
    # 稍微增强初始偏置，确保稳妥落入目标吸引子盆地
    v = 2.0 * np.random.rand(N) + 4.0 * patterns[target_pattern_idx]
    v_hist = np.zeros((steps, N))
    
    for i in range(steps):
        v_hist[i, :] = v
        # 【核心修正】：增强全局抑制至 -19.0，确保静息单元的输入 I < -20
        I = W @ v - 16.0 
        v = v + (dt / tau) * (-v + F(I))
    return v_hist

# 运行模拟：分别提取模式 0 和 模式 1
v_hist_B = simulate(0)
v_hist_C = simulate(1)

# ================= 绘图代码 =================
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 2])

# 面板 A: 代表性神经元
axA1 = plt.subplot(gs[0, 0])
axA2 = plt.subplot(gs[1, 0])

# 对于模式 0 (17-30为1)，20号是活跃的，10号是静息的
active_neuron = 20
inactive_neuron = 10
axA1.plot(time, v_hist_B[:, active_neuron], 'k')
axA2.plot(time, v_hist_B[:, inactive_neuron], 'k')

axA1.set_ylabel(r'$v$ (Hz)', fontsize=14)
axA1.set_xticklabels([])
axA1.set_ylim(bottom=0) 

axA2.set_ylabel(r'$v$ (Hz)', fontsize=14)
axA2.set_xlabel(r'$t$ (ms)', fontsize=14)
axA2.set_ylim(bottom=0)

# 动态归一化绘图函数
def plot_raster_thickness(ax, v_history, title):
    max_v = np.max(v_history)
    if max_v == 0: 
        max_v = 1.0 
        
    for i in range(N):
        thickness = 0.35 * (v_history[:, i] / max_v) 
        ax.fill_between(time, i - thickness, i + thickness, color='k', lw=0)
    
    ax.set_ylim(-1, N)
    ax.set_ylabel('neurons', fontsize=14)
    ax.set_xlabel(r'$t$ (ms)', fontsize=14)
    ax.set_title(title, loc='left', fontsize=16, fontweight='bold')
    ax.invert_yaxis()

# 面板 B & C
axB = plt.subplot(gs[:, 1])
plot_raster_thickness(axB, v_hist_B, 'B')

axC = plt.subplot(gs[:, 2])
plot_raster_thickness(axC, v_hist_C, 'C')

plt.tight_layout()
plt.show()