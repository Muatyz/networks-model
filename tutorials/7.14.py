# 非线性网络
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 模型与网络参数设置
# ==========================================
N = 360                      # 神经元数量
theta = np.linspace(-180, 180, N, endpoint=False) # 角度 (-180 到 180度)
theta_rad = np.deg2rad(theta) # 转换为弧度用于计算

tau = 10.0                   # 神经元时间常数 (ms)
dt = 0.5                     # 时间步长 (ms)

# 突触连接矩阵 W (平移不变性: 只依赖于角度差)
# 使用余弦形式: 局部兴奋 (J1 > 0) + 广泛抑制 (J0 < 0)
J0 = -2.0
J1 = 4.0
W = np.zeros((N, N))
for i in range(N):
    W[i, :] = J0 + J1 * np.cos(theta_rad[i] - theta_rad)
W = W / N  # 归一化

# 激活函数: 带有阈值的线性函数 (ReLU)
def f_rect(x):
    return np.maximum(0, x)

# ==========================================
# 2. 模拟网络动力学
# ==========================================

# --- 阶段 1: 带有局部峰值的输入 (对应 Panel A 和 B) ---
# 输入 h = 恒定背景 + 高斯局部刺激
h_A = 2.0 + 2.5 * np.exp(-0.5 * (theta_rad / 0.4)**2)

v = np.zeros(N) # 初始活动为 0
# 运行一段时间到达稳态
for _ in range(int(200 / dt)): 
    total_input = h_A + W @ v
    dv = (-v + f_rect(total_input)) / tau * dt
    v += dv
v_B = v.copy()  # 记录阶段 1 的稳态活动

# --- 阶段 2: 撤销局部刺激，变为完全平坦的恒定输入 (对应 Panel C 和 D) ---
# 输入 h = 恒定背景
h_C = np.full(N, 2.0)

# 从 v_B 的状态继续模拟
for _ in range(int(300 / dt)): 
    total_input = h_C + W @ v
    dv = (-v + f_rect(total_input)) / tau * dt
    v += dv
v_D = v.copy()  # 记录阶段 2 的稳态活动 (工作记忆)


# ==========================================
# 3. 绘制结果 (严格按照原书图 7.14 的排版)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
plt.rcParams['font.size'] = 12

# (A) 带有局部峰值的输入
axs[0, 0].plot(theta, h_A, 'k-', linewidth=1.5)
axs[0, 0].set_ylim(-5, 8)
axs[0, 0].set_yticks([-5, 0, 5])
axs[0, 0].set_ylabel(r'$h$', fontsize=14, rotation=0, labelpad=15)
axs[0, 0].text(-0.1, 1.05, 'A', transform=axs[0, 0].transAxes, fontsize=16, fontweight='bold')

# (B) 局部输入导致的稳态网络活动
axs[0, 1].plot(theta, v_B, 'k-', linewidth=1.5)
axs[0, 1].set_ylim(0, 25)
axs[0, 1].set_yticks([0, 5, 10, 15, 20])
axs[0, 1].set_ylabel(r'$v$ (Hz)', fontsize=14)
axs[0, 1].text(-0.1, 1.05, 'B', transform=axs[0, 1].transAxes, fontsize=16, fontweight='bold')

# (C) 恒定的平坦输入
axs[1, 0].plot(theta, h_C, 'k-', linewidth=1.5)
axs[1, 0].set_ylim(-5, 8)
axs[1, 0].set_yticks([-5, 0, 5])
axs[1, 0].set_ylabel(r'$h$', fontsize=14, rotation=0, labelpad=15)
axs[1, 0].text(-0.1, 1.05, 'C', transform=axs[1, 0].transAxes, fontsize=16, fontweight='bold')

# (D) 恒定输入下维持的网络活动 (记忆)
axs[1, 1].plot(theta, v_D, 'k-', linewidth=1.5)
axs[1, 1].set_ylim(0, 25)
axs[1, 1].set_yticks([0, 5, 10, 15, 20])
axs[1, 1].set_ylabel(r'$v$ (Hz)', fontsize=14)
axs[1, 1].text(-0.1, 1.05, 'D', transform=axs[1, 1].transAxes, fontsize=16, fontweight='bold')

# 统一设置 X 轴
for ax in axs.flat:
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(r'$\theta$ (deg)', fontsize=14)
    # 去除顶部和右侧的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=3.0)
plt.show()