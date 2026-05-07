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
# 2. 生成多峰外部输入 h(theta)
# 构造两个峰值，分别在 -90度和 +90度。
# 按照图 7.12A，-90度的峰稍微高一些。这里我们用高斯函数来构造。
# ==========================================
# 峰 1: 位于 -90 度，振幅约 5.0
peak1 = 5.0 * np.exp(-0.5 * ((theta + 90) / 20.0)**2)
# 峰 2: 位于 +90 度，振幅约 3.0
peak2 = 3.0 * np.exp(-0.5 * ((theta - 90) / 20.0)**2)

h_input = peak1 + peak2

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
    # 计算傅里叶级数的投影，这代表了余弦权重的全连接网络
    v_cos = np.trapezoid(v * np.cos(theta_rad), dx=dtheta_rad) / np.pi
    v_sin = np.trapezoid(v * np.sin(theta_rad), dx=dtheta_rad) / np.pi
    
    # 循环输入
    recurrent_input = lambda_1 * (v_cos * np.cos(theta_rad) + v_sin * np.sin(theta_rad))
    
    # 核心：计算总输入并进行半波整流 (引入非线性竞争)
    total_input = h_input + recurrent_input
    firing_rate_target = np.maximum(0, total_input)
    
    # 更新放电率 (ODE)
    dv = -v + firing_rate_target
    v = v + (dt / tau) * dv

# ==========================================
# 4. 绘图 (复刻 Figure 7.12)
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
title_fs, label_fs = 12, 10

# A: 双峰输入
axs[0].plot(theta, h_input, color='black', linewidth=1.5)
axs[0].set_title('A. Input $h(\\theta)$', loc='left', fontsize=title_fs, fontweight='bold')
axs[0].set_xlabel(r'$\theta$ (deg)', fontsize=label_fs)
axs[0].set_ylabel('$h$', fontsize=label_fs, rotation=0, labelpad=10)
axs[0].set_xlim([-180, 180])
axs[0].set_ylim([-5, 6])
axs[0].set_xticks([-180, -90, 0, 90, 180])

# B: 单峰输出 (赢者通吃结果)
axs[1].plot(theta, v, color='black', linewidth=1.5)
axs[1].set_title('B. Output $v_\\infty(\\theta)$', loc='left', fontsize=title_fs, fontweight='bold')
axs[1].set_xlabel(r'$\theta$ (deg)', fontsize=label_fs)
axs[1].set_ylabel('$v$ (Hz)', fontsize=label_fs, rotation=0, labelpad=15)
axs[1].set_xlim([-180, 180])
axs[1].set_ylim([0, 80])
axs[1].set_xticks([-180, -90, 0, 90, 180])

plt.tight_layout()
plt.show()