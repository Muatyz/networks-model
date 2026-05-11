import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('./tutorial.mplstyle')

# --- 1. 参数设置 ---
M_EE = 1.25
M_IE = 1.0
M_II = 0.0
M_EI = -1.0
tau_E = 10.0  # 单位: ms

# 设置全局字体为类似 LaTeX 的衬线字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# 创建画布与网格布局
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.4, wspace=0.3)

# ==========================================
# 图 A: 相平面与零倾线 (Phase Plane & Nullclines)
# ==========================================
ax_A = fig.add_subplot(gs[:, 0])

# 定义 v_E 的范围
v_E = np.linspace(0, 60, 400)

# 计算零倾线
# dv_E/dt = 0 -> v_I = 0.25 * v_E + 10
v_I_E_null = 0.25 * v_E + 10
# dv_I/dt = 0 -> v_I = v_E - 10
v_I_I_null = v_E - 10

# 绘制零倾线
ax_A.plot(v_E, v_I_E_null, 'k-', lw=1.5)
ax_A.plot(v_E, v_I_I_null, 'k-', lw=1.5)

# 绘制并标注固定点
v_E_fp, v_I_fp = 26.67, 16.67
ax_A.plot(v_E_fp, v_I_fp, 'ko', markersize=9, zorder=5)

# 绘制相平面的流动方向箭头 (使用 annotate 画箭头线条)
arrow_props = dict(arrowstyle='-|>', color='k', lw=1.2, shrinkA=0, shrinkB=0)
# 下方水平向右
ax_A.annotate('', xy=(42, 4.5), xytext=(8, 4.5), arrowprops=arrow_props)
# 右侧垂直向上
ax_A.annotate('', xy=(42, 24), xytext=(42, 4.5), arrowprops=arrow_props)
# 上方水平向左
ax_A.annotate('', xy=(8, 25.5), xytext=(42, 25.5), arrowprops=arrow_props)
# 左侧垂直向下
ax_A.annotate('', xy=(8, 5.5), xytext=(8, 25.5), arrowprops=arrow_props)

# 添加文本标注
ax_A.text(44, 23, r'$dv_E/dt = 0$', fontsize=15, va='center')
ax_A.text(28, 28, r'$\leftarrow dv_I/dt = 0$', fontsize=15, va='center')
ax_A.text(-6, 34, 'A', fontsize=16, fontweight='bold')

# 样式美化
ax_A.set_xlim(0, 60)
ax_A.set_ylim(0, 32)
ax_A.set_xlabel(r'$v_E$ (Hz)', fontsize=15)
ax_A.set_ylabel(r'$v_I$ (Hz)', fontsize=15)
ax_A.spines['top'].set_visible(False)
ax_A.spines['right'].set_visible(False)
ax_A.tick_params(direction='in', labelsize=12)


# ==========================================
# 计算特征值 (Eigenvalues)
# ==========================================
tau_I = np.linspace(1, 100, 500)

term1 = (M_EE - 1) / tau_E
term2 = (M_II - 1) / tau_I

# 计算迹 (Trace) 相关的部分，并转换为 s^-1
trace_part = (term1 + term2) * 1000 / 2

# 计算根号内部 (判别式)
sqrt_inner = (term1 - term2)**2 + 4 * M_EI * M_IE / (tau_E * tau_I)

# 使用复数域求根，避免负数开方报错，并转换为 s^-1
sqrt_part = np.sqrt(sqrt_inner.astype(complex)) * 1000 / 2

# 取具有较大实部的特征值分支
eigenval = trace_part + sqrt_part


# ==========================================
# 图 B 上半部分: 特征值实部 Re{lambda}
# ==========================================
ax_B_real = fig.add_subplot(gs[0, 1])
ax_B_real.plot(tau_I, np.real(eigenval), 'k-', lw=1.5)

# 样式美化: 将 x 轴移动到 y=0 的位置
ax_B_real.spines['bottom'].set_position('zero')
ax_B_real.spines['top'].set_visible(False)
ax_B_real.spines['right'].set_visible(False)

ax_B_real.set_xlim(0, 100)
ax_B_real.set_ylim(-45, 20)
ax_B_real.set_yticks([-40, -20, 20])
ax_B_real.tick_params(direction='in', labelsize=12)

# 手动调整 ylabel 和 xlabel 位置以适应移动后的脊背
ax_B_real.set_ylabel(r'$Re\{\lambda\}$ (s$^{-1}$)', fontsize=15)
ax_B_real.text(80, -15, r'$\tau_I$ (ms)', fontsize=15)
ax_B_real.text(-15, 25, 'B', fontsize=16, fontweight='bold')


# ==========================================
# 图 B 下半部分: 特征值虚部 Im{lambda}/2pi (频率 Hz)
# ==========================================
ax_B_imag = fig.add_subplot(gs[1, 1])

# 计算频率 (Hz)，虚部本身是角频率 (rad/s)
freq = np.imag(eigenval) / (2 * np.pi)
ax_B_imag.plot(tau_I, freq, 'k-', lw=1.5)

# 样式美化
ax_B_imag.set_xlim(0, 100)
ax_B_imag.set_ylim(0, 15)
ax_B_imag.set_yticks([0, 4, 8, 12])
ax_B_imag.set_xlabel(r'$\tau_I$ (ms)', fontsize=15)
ax_B_imag.set_ylabel(r'$Im\{\lambda\}/2\pi$ (Hz)', fontsize=15)
ax_B_imag.spines['top'].set_visible(False)
ax_B_imag.spines['right'].set_visible(False)
ax_B_imag.tick_params(direction='in', labelsize=12)

plt.show()