import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 网络与仿真参数设置
# ==========================================
N = 64                          # 神经元数量
theta = np.linspace(-90, 90, N, endpoint=False) # 偏好角度空间 [-90, 90)
tau = 10.0                      # 神经元时间常数 (ms)
dt = 1.0                        # 欧拉法仿真步长 (ms)
T_total = 400                   # 总仿真时长 (ms)
time_steps = int(T_total / dt)

# ==========================================
# 2. 构建 Gabor-like 循环权重矩阵 W
# ==========================================
def angle_diff(t1, t2):
    """计算环形空间中的最短角度差 [-90, 90)"""
    diff = t1 - t2
    return (diff + 90) % 180 - 90

A = 1.2            # 兴奋强度
sigma_w = 15.0     # 兴奋扩散范围 (度)
k = np.pi / 45.0   # 空间频率 (控制长程抑制的出现点)
B = 0.5            # 全局抑制

W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d_theta = angle_diff(theta[i], theta[j])
        W[i, j] = A * np.exp(-(d_theta**2) / (2 * sigma_w**2)) * np.cos(k * d_theta) - B

# 缩放权重以确保网络动力学稳定
W = W / N * 20.0 

# ==========================================
# 3. 构建输入信号 h(theta, t)
# ==========================================
true_theta = 10.0      # 真实的刺激角度
h_baseline = 10.0      # 恒定的基线输入
signal_strength = 1.0  # 微弱的真实信号隆起
noise_std = 0.25       # 高频噪声的标准差

h = np.zeros((N, time_steps))

# 阶段 1 (0-100ms)：带噪初始输入
initial_signal = h_baseline + signal_strength * np.exp(-(angle_diff(theta, true_theta)**2) / (2 * 10.0**2))
for t in range(int(100 / dt)):
    h[:, t] = initial_signal + np.random.normal(0, noise_std, N)

# 阶段 2 (100ms 之后)：恒定基线输入 (维持吸引子)
for t in range(int(100 / dt), time_steps):
    h[:, t] = h_baseline

# ==========================================
# 4. 运行网络动力学仿真
# ==========================================
v = np.zeros((N, time_steps))

def relu(x):
    return np.maximum(0, x)

# 欧拉法积分
for t in range(time_steps - 1):
    synaptic_input = np.dot(W, v[:, t])
    dv = (-v[:, t] + relu(synaptic_input + h[:, t])) / tau
    v[:, t+1] = v[:, t] + dv * dt

# ==========================================
# 5. 结果可视化 (动态自适应 Y 轴)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

# 图 A: 带噪初始输入 (取第 50 步的快照)
ax1.scatter(theta, h[:, 50], color='black', s=15)
ax1.set_title('A: Noisy initial inputs $h(\\theta)$')
ax1.set_xlabel('$\\theta$ (deg)')
ax1.set_ylabel('$h$')

# 动态计算图 A 的 Y 轴范围 (留出 5% 的上下边距)
h_min, h_max = np.min(h[:, 50]), np.max(h[:, 50])
h_margin = (h_max - h_min) * 0.1 if h_max > h_min else 1.0
ax1.set_ylim(h_min - h_margin, h_max + h_margin)
ax1.set_xticks([-90, -45, 0, 45, 90])

# 图 B: 重编码后的平滑群体活动
ax2.scatter(theta, v[:, -1], color='black', s=15)
ax2.set_title('B: Smoothed activity $v(\\theta)$')
ax2.set_xlabel('$\\theta$ (deg)')
ax2.set_ylabel('$v$ (Hz)')

# 动态计算图 B 的 Y 轴范围 (从 0 开始，顶部留出 10% 的空间展示峰值)
v_max = np.max(v[:, -1])
# 防止初始状态全为 0 导致 ylim(0, 0) 报错
v_ymax = v_max * 1.1 if v_max > 0 else 1.0 
ax2.set_ylim(0, v_ymax)
ax2.set_xticks([-90, -45, 0, 45, 90])

# 去除顶部和右侧的边框以模仿原书排版风格
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()