import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ---------------------------------------------------------
# 1. 参数设置 (根据教材 Figure 7.10 的描述)
# ---------------------------------------------------------
tau_r = 10.0      # 时间常数 (ms)，设定为10ms作为典型值
lambda_0 = 7.3    # 循环抑制强度
lambda_1 = 11.0   # 循环兴奋强度
A = 40.0          # 前馈输入的基础幅度 (Hz)
epsilon = 0.1     # 输入的方位角调制强度
contrasts = [0.8, 0.4, 0.2, 0.1]  # 4个对比度水平: 80%, 40%, 20%, 10%

N = 100           # 神经元的数量 (离散化角度)
theta = np.linspace(-np.pi/2, np.pi/2, N, endpoint=False) # 角度范围从 -90度到90度
d_theta = np.pi / N

# ---------------------------------------------------------
# 2. 构建循环连接权重矩阵 W
# W(theta - theta') = -lambda_0 + lambda_1 * cos(2(theta - theta'))
# ---------------------------------------------------------
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        W[i, j] = (-lambda_0 + lambda_1 * np.cos(2 * (theta[i] - theta[j])))

# ---------------------------------------------------------
# 3. 定义仿真函数
# ---------------------------------------------------------
def simulate_network(c, num_steps=500, dt=1.0):
    # 构建前馈输入 h(theta)
    # h(theta) = A * c * (1 - epsilon + epsilon * cos(2 * theta))
    h = A * c * (1 - epsilon + epsilon * np.cos(2 * theta))
    
    # 初始放电率设为0
    v = np.zeros(N)
    
    # 欧拉法数值积分求解常微分方程 (Eq 7.36)
    for _ in range(num_steps):
        # 计算积分项 (矩阵乘法模拟积分)
        recurrent_input = (1.0 / np.pi) * np.dot(W, v) * d_theta
        
        # 计算总驱动力
        total_input = h + recurrent_input
        
        # 经过 ReLU 激活函数 [x]+
        firing_rate_steady = np.maximum(0, total_input)
        
        # 更新 dv/dt
        dv_dt = (-v + firing_rate_steady) / tau_r
        v = v + dv_dt * dt
        
    return h, v

# ---------------------------------------------------------
# 4. 运行仿真并绘图 (复现 Fig 7.10)
# ---------------------------------------------------------
# 缩小 figsize 以避免占用过多篇幅
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

# 将弧度转换为角度用于绘图
theta_deg = theta * 180 / np.pi

# 存储结果
h_results = []
v_results = []

for c in contrasts:
    h, v = simulate_network(c=c)
    h_results.append(h)
    v_results.append(v)
    
    # 绘制前馈输入 h(theta)
    ax1.plot(theta_deg, h, label=f'Contrast: {c*100:.0f}%', color='black', alpha=c+0.2)
    # 绘制网络输出 v(theta)
    ax2.plot(theta_deg, v, label=f'Contrast: {c*100:.0f}%', color='black', alpha=c+0.2)

# 图表 A 美化
ax1.set_title('A: Feedforward Input (h)')
ax1.set_xlabel(r'$\theta$ (deg)')  # 使用 r前缀 修复 LaTeX 编译
ax1.set_ylabel('h (Hz)')
ax1.set_xlim([-40, 40])
ax1.set_ylim([0, 35])

# 图表 B 美化
ax2.set_title('B: Network Output Firing Rate (v)')
ax2.set_xlabel(r'$\theta$ (deg)')  # 使用 r前缀 修复 LaTeX 编译
ax2.set_ylabel('v (Hz)')
ax2.set_xlim([-40, 40])
ax2.set_ylim([0, 80])
ax2.legend()

# 使用正确的函数调整子图布局
plt.tight_layout()
plt.show()