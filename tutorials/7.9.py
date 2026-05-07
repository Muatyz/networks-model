import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 初始化网络空间与参数
# ==========================================
theta = np.linspace(-180, 180, 200)
theta_rad = np.radians(theta)
dtheta_rad = theta_rad[1] - theta_rad[0]

# ==========================================
# 2. 生成外部输入 h(theta)
# ==========================================
np.random.seed(42)
# 为了配合非线性网络，我们稍微提高基础余弦的振幅，使其能突破阈值
base_cosine = 2.0 * np.cos(theta_rad) 
noise = 1.5 * np.random.randn(200)
h_input = base_cosine + noise

# ==========================================
# 3. 非线性动力学演化 (Euler Method)
# ==========================================
lambda_1 = 1.9
tau = 20.0 # 时间常数 (ms)
dt = 0.5   # 步长 (ms)
steps = 500

v = np.zeros_like(theta_rad) # 初始放电率均为 0

# 模拟网络随时间的动态演化
for step in range(steps):
    # 利用卷积的特性高效计算循环反馈输入 M * v
    # 因为 M 只有余弦成分，积分化简为计算 v 的傅里叶余弦和正弦系数
    v_cos = np.trapezoid(v * np.cos(theta_rad), dx=dtheta_rad) / np.pi
    v_sin = np.trapezoid(v * np.sin(theta_rad), dx=dtheta_rad) / np.pi
    
    # 循环输入
    recurrent_input = lambda_1 * (v_cos * np.cos(theta_rad) + v_sin * np.sin(theta_rad))
    
    # 核心：计算总输入并进行半波整流 (ReLU: np.maximum)
    total_input = h_input + recurrent_input
    firing_rate_target = np.maximum(0, total_input)
    
    # 更新放电率 (ODE)
    dv = -v + firing_rate_target
    v = v + (dt / tau) * dv

# ==========================================
# 4. 傅里叶分析
# ==========================================
mu_values = np.arange(10)
h_amplitudes = np.zeros(10)
v_amplitudes = np.zeros(10)

for mu in mu_values:
    # 输入的傅里叶振幅
    if mu == 0:
        h_cos = np.trapezoid(h_input, dx=dtheta_rad) / (2 * np.pi)
        h_sin = 0
        v_cos = np.trapezoid(v, dx=dtheta_rad) / (2 * np.pi)
        v_sin = 0
    else:
        h_cos = np.trapezoid(h_input * np.cos(mu * theta_rad), dx=dtheta_rad) / np.pi
        h_sin = np.trapezoid(h_input * np.sin(mu * theta_rad), dx=dtheta_rad) / np.pi
        v_cos = np.trapezoid(v * np.cos(mu * theta_rad), dx=dtheta_rad) / np.pi
        v_sin = np.trapezoid(v * np.sin(mu * theta_rad), dx=dtheta_rad) / np.pi
        
    h_amplitudes[mu] = np.sqrt(h_cos**2 + h_sin**2)
    v_amplitudes[mu] = np.sqrt(v_cos**2 + v_sin**2)

# ==========================================
# 5. 绘图 (完美复刻 Figure 7.9)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
title_fs, label_fs, tick_fs = 12, 10, 9

# A: Input
axs[0, 0].plot(theta, h_input, color='black', linewidth=1.2)
axs[0, 0].set_title('A. Input $h(\\theta)$', loc='left', fontsize=title_fs, fontweight='bold')
axs[0, 0].set_ylabel('$h$', fontsize=label_fs, rotation=0, labelpad=10)
axs[0, 0].set_xticks([-180, -90, 0, 90, 180])
axs[0, 0].grid(True, linestyle='--', alpha=0.5)

# B: Output (现在是一个非线性的 Bump!)
axs[0, 1].plot(theta, v, color='black', linewidth=1.5)
axs[0, 1].set_title('B. Nonlinear Output $v_\\infty(\\theta)$', loc='left', fontsize=title_fs, fontweight='bold')
axs[0, 1].set_ylabel('$v$', fontsize=label_fs, rotation=0, labelpad=10)
axs[0, 1].set_xticks([-180, -90, 0, 90, 180])
axs[0, 1].grid(True, linestyle='--', alpha=0.5)

# C & D: Fourier Amplitudes
axs[1, 0].bar(mu_values, h_amplitudes, color='black', width=0.8)
axs[1, 0].set_title('C. Fourier Amplitudes (Input)', loc='left', fontsize=title_fs, fontweight='bold')
axs[1, 0].set_xticks(mu_values)

axs[1, 1].bar(mu_values, v_amplitudes, color='black', width=0.8)
axs[1, 1].set_title('D. Fourier Amplitudes (Output)', loc='left', fontsize=title_fs, fontweight='bold')
axs[1, 1].set_xticks(mu_values)

plt.tight_layout()
plt.show()