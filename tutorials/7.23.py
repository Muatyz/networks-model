import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 网络参数与时间设置
# ==========================================
N = 180  # 离散化神经元数量，覆盖 -90° 到 90°
theta_deg = np.linspace(-90, 90, N, endpoint=False)
theta = np.deg2rad(theta_deg)

tau_E = 10.0  # ms
tau_I = 10.0  # ms
dt = 0.05     # ms, 仿真步长 (足够小以保证稳定性)
T = 500.0     # ms, 仿真总时长
time = np.arange(0, T, dt)
steps = len(time)

# ==========================================
# 2. 权重矩阵构建
# ==========================================
# 距离矩阵 dtheta
dtheta = theta[:, None] - theta[None, :]

# M_EE: [5.9 + 7.8cos(2(θ-θ'))]+ / N
W_EE = np.maximum(0, 5.9 + 7.8 * np.cos(2 * dtheta)) / N

# M_IE: 均匀连接，所有值为 13.3 / N
W_IE = np.ones((N, N)) * 13.3 / N

# M_EI: 局部自抑制，等效为对角线为 1 的对角矩阵
W_EI = - np.eye(N)

# M_II = 0
h_I = 0.0

# ==========================================
# 3. 外部输入信号构建
# ==========================================
np.random.seed(42)  # 固定随机种子以稳定复现
noise = np.random.normal(0, 0.4, N)

# 调制(Tuned)与未调制(Untuned)输入 (噪声随空间固定，不随时间变化)
h_E_tuned = 8.0 + 5.0 * np.cos(2 * theta) + noise
h_E_untuned = 8.0 + noise

# ==========================================
# 4. 核心仿真函数 (基于 ReLU 激活)
# ==========================================
def simulate(h_E):
    v_E = np.zeros((steps, N))
    v_I = np.zeros((steps, N))
    
    for i in range(1, steps):
        # 计算总输入，np.dot / @ 实现了离散积分(求和)
        I_E = h_E + W_EE @ v_E[i-1] + W_EI @ v_I[i-1]
        I_I = h_I + W_IE @ v_E[i-1]
        
        # 欧拉法更新，使用 [x]+ 即 np.maximum(0, x) 限制发放率非负
        dv_E = (-v_E[i-1] + np.maximum(0, I_E)) / tau_E
        dv_I = (-v_I[i-1] + np.maximum(0, I_I)) / tau_I
        
        v_E[i] = v_E[i-1] + dt * dv_E
        v_I[i] = v_I[i-1] + dt * dv_I
        
    return v_E, v_I

# 运行仿真
v_E_tuned, _ = simulate(h_E_tuned)
v_E_untuned, _ = simulate(h_E_untuned)

# ==========================================
# 5. 绘图代码
# ==========================================
# 找到 0° 和 -37° 对应的神经元索引
idx_0 = np.argmin(np.abs(theta_deg - 0))
idx_37 = np.argmin(np.abs(theta_deg - (-37)))

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

def format_ax(ax):
    """去除上方和右侧的边框以贴合原书图表风格"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)

# --------- 图 A：时间平均响应 ---------
avg_tuned = np.mean(v_E_tuned, axis=0)
avg_untuned = np.mean(v_E_untuned, axis=0)

axs[0].plot(theta_deg, avg_tuned, 'k-', linewidth=1.2, label='Tuned')
axs[0].plot(theta_deg, avg_untuned, 'k--', linewidth=1.2, label='Untuned')
axs[0].plot(0, avg_tuned[idx_0], 'ko', markerfacecolor='none', markersize=7)
axs[0].plot(-37, avg_tuned[idx_37], 'kx', markersize=7)

axs[0].set_xlim(-90, 90)
axs[0].set_xticks([-90, -45, 0, 45, 90])
axs[0].set_ylim(0, 200)
axs[0].set_xlabel(r'$\theta$ (deg)', fontsize=12)
axs[0].set_ylabel(r'average $v$ (Hz)', fontsize=12)
axs[0].set_title('A', loc='left', fontweight='bold', fontsize=14)
format_ax(axs[0])

# --------- 图 B：Tuned输入下含时输出 ---------
marker_interval = int(25 / dt) # 每隔一定的仿真步数画一个标记点，防止密集
axs[1].plot(time, v_E_tuned[:, idx_0], 'k-', linewidth=0.8, 
            marker='o', markerfacecolor='none', markevery=marker_interval)
axs[1].plot(time, v_E_tuned[:, idx_37], 'k--', linewidth=0.8, 
            marker='x', markevery=marker_interval)

axs[1].set_xlim(0, 500)
axs[1].set_xticks([0, 250, 500])
axs[1].set_ylim(0, 1000)
axs[1].set_xlabel('time (ms)', fontsize=12)
axs[1].set_ylabel(r'$v$ (Hz)', fontsize=12)
axs[1].set_title('B', loc='left', fontweight='bold', fontsize=14)
format_ax(axs[1])

# --------- 图 C：Untuned输入下含时输出 ---------
axs[2].plot(time, v_E_untuned[:, idx_0], 'k-', linewidth=0.8, 
            marker='o', markerfacecolor='none', markevery=marker_interval)
axs[2].plot(time, v_E_untuned[:, idx_37], 'k--', linewidth=0.8, 
            marker='x', markevery=marker_interval)

axs[2].set_xlim(0, 500)
axs[2].set_xticks([0, 250, 500])
axs[2].set_ylim(0, 1000)
axs[2].set_xlabel('time (ms)', fontsize=12)
axs[2].set_title('C', loc='left', fontweight='bold', fontsize=14)
format_ax(axs[2])

plt.tight_layout()
plt.show()