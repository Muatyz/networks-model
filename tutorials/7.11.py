import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./tutorial.mplstyle')

# ---------------------------------------------------------
# 1. 参数设置 (复现 Fig 7.11)
# ---------------------------------------------------------
tau_r = 10.0      
lambda_1 = 0.95   # 接近1的全局兴奋系数
N = 300           
phi = np.linspace(-np.pi, np.pi, N, endpoint=False) # 空间相位从 -180 到 180 度
d_phi = 2 * np.pi / N

# ---------------------------------------------------------
# 2. 构建输入 h(phi) (模拟图 7.11A 的简单细胞响应)
# 构造一个带有基线的钟形曲线
# 这里我们用半波整流的余弦函数来近似简单细胞的调谐，并调整振幅使其最大值为30
# ---------------------------------------------------------
h = 30.0 * np.maximum(0, np.cos(phi)) 

# ---------------------------------------------------------
# 3. 运行网络动力学仿真 (Eq 7.38)
# ---------------------------------------------------------
v = np.zeros(N)
dt = 1.0
num_steps = 1000

for _ in range(num_steps):
    # 计算积分项：均匀连接，相当于对整个 v(phi') 求和后再乘以 lambda_1 / (2*pi)
    # \int v(\phi') d\phi' \approx sum(v) * d_phi
    recurrent_input = (lambda_1 / (2 * np.pi)) * np.sum(v) * d_phi
    
    # 总输入
    total_input = h + recurrent_input
    
    # ReLU 激活 [ ]+
    firing_rate_steady = np.maximum(0, total_input)
    
    # 更新状态
    dv_dt = (-v + firing_rate_steady) / tau_r
    v = v + dv_dt * dt

# ---------------------------------------------------------
# 4. 绘图展示
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

phi_deg = phi * 180 / np.pi

# 图 A: 前馈输入
ax1.plot(phi_deg, h, color='black')
ax1.set_title(r'A: Feedforward Input $h(\phi)$')
ax1.set_xlabel(r'$\phi$ (deg)')
ax1.set_ylabel('h (Hz)')
ax1.set_xlim([-180, 180])
ax1.set_ylim([0, 35])
ax1.set_xticks([-180, -90, 0, 90, 180])

# 图 B: 网络稳态输出
ax2.plot(phi_deg, v, color='black')
ax2.set_title(r'B: Network Response $v(\phi)$')
ax2.set_xlabel(r'$\phi$ (deg)')
ax2.set_ylabel('v (Hz)')
ax2.set_xlim([-180, 180])
margin = 5
ax2.set_ylim([0, np.max(v) + margin])
ax2.set_xticks([-180, -90, 0, 90, 180])

plt.tight_layout()
plt.show()