import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 修正后的网络参数与物理量 (连续吸引子状态)
# ==========================================
N = 180                     
theta = np.linspace(-180, 180, N, endpoint=False)  
dtheta = 360 / N            
theta_rad = np.deg2rad(theta) 

tau = 10.0                  
dt = 0.5                    
T = 200                     # 稍微延长模拟时间，确保强循环网络达到稳态

# 将网络推入吸引子状态 (Attractor Regime)
J0 = 5.0                    # 增强全局抑制，防止网络兴奋过度爆炸
J1 = 12.0                   # 大幅增强局部兴奋，让循环连接主导形状
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        diff = theta_rad[i] - theta_rad[j]
        W[i, j] = (-J0 + J1 * np.cos(diff)) / (2 * np.pi)

def F(x):
    return np.maximum(0, x)

# ==========================================
# 2. 外部输入 h(theta)
# ==========================================
sigma_in = 45.0             # 拓宽外部输入，不再强迫网络变得那么窄
A_in = 1.0                  # 减弱前馈输入的强度，让它仅仅作为"触发器"
h_base = A_in * np.exp(- (theta)**2 / (2 * sigma_in**2))

c_values = [-0.5, 0.0, 0.5] # 缩小加性常数的扰动范围，以防破坏网络稳态

# ==========================================
# 3. 运行网络动力学模拟
# ==========================================
v_steady_states = []        # 存储稳态放电率
h_inputs = []               # 存储对应的总输入

for c in c_values:
    # 构造总输入：视觉峰值 + 注视方向常数
    h = h_base + c
    h_inputs.append(h)
    
    # 初始化网络所有神经元的放电率为 0
    v = np.zeros(N)
    
    # 欧拉法求解微分方程
    for t in np.arange(0, T, dt):
        # 计算所有神经元传来的循环输入 (即积分项 W * v * dtheta)
        recurrent_input = W @ v * np.deg2rad(dtheta)
        
        # 计算放电率的导数 dv/dt
        dv = (-v + F(h + recurrent_input)) / tau
        
        # 更新放电率 v
        v = v + dv * dt
        
    v_steady_states.append(v)

# ==========================================
# 4. 绘图 (复现图 7.13)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制图 A：网络输入 h
for h in h_inputs:
    ax1.plot(theta, h, color='k')
ax1.set_title('(A) Network Input $h(\\theta)$', fontsize=14)
ax1.set_xlabel('$\\theta$ (deg)', fontsize=12)
ax1.set_ylabel('$h$', fontsize=12)
ax1.set_xlim(-180, 180)
ax1.set_ylim(-1, 2)
ax1.set_xticks([-180, -90, 0, 90, 180])

# 绘制图 B：网络输出的稳态放电率 v
for v in v_steady_states:
    ax2.plot(theta, v, color='k')
ax2.set_title('(B) Network Output $v(\\theta)$', fontsize=14)
ax2.set_xlabel('$\\theta$ (deg)', fontsize=12)
ax2.set_ylabel('$v$ (Hz)', fontsize=12)
ax2.set_xlim(-180, 180)
ax2.set_xticks([-180, -90, 0, 90, 180])

plt.tight_layout()
plt.show()