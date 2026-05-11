import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.style.use('./tutorial.mplstyle')

# --- 1. 参数设置 ---
M_EE = 1.25
M_IE = 1.0
M_II = 0.0
M_EI = -1.0
gamma_E = -10.0
gamma_I = 10.0
tau_E = 10.0  # 单位: ms
tau_I = 30.0  # 单位: ms (图 7.18 设为 30 ms，此时固定点稳定)

# --- 2. 定义动力学方程 ---
def firing_rate_model(v, t):
    v_E, v_I = v
    # 阈值线性响应函数 [x]_+ = max(x, 0)
    input_E = M_EE * v_E + M_EI * v_I - gamma_E
    input_I = M_II * v_I + M_IE * v_E - gamma_I
    
    rate_E = max(input_E, 0)
    rate_I = max(input_I, 0)
    
    dvE_dt = (-v_E + rate_E) / tau_E
    dvI_dt = (-v_I + rate_I) / tau_I
    
    return [dvE_dt, dvI_dt]

# --- 3. 运行数值模拟 ---
# 时间轴：从 0 到 1000 ms
t = np.linspace(0, 1000, 2000)
# 初始条件 (估算自图中空心圆位置)
v0 = [51.0, 29.5]
# 解 ODE
solution = odeint(firing_rate_model, v0, t)
v_E_t = solution[:, 0]
v_I_t = solution[:, 1]

# --- 4. 绘图 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# 创建适配 Notebook 的紧凑画布
fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.3)

# ==========================================
# 图 A: 随时间演化的放电率
# ==========================================
ax_A_top = fig.add_subplot(gs[0, 0])
ax_A_bot = fig.add_subplot(gs[1, 0])

# 绘制 v_E(t)
ax_A_top.plot(t, v_E_t, 'k-', lw=1.2)
ax_A_top.set_xlim(0, 1000)
ax_A_top.set_ylim(0, 60)
ax_A_top.set_yticks([0, 20, 40, 60])
ax_A_top.set_ylabel(r'$v_E$ (Hz)', fontsize=12)
ax_A_top.text(-200, 65, 'A', fontsize=14, fontweight='bold', va='center')

# 绘制 v_I(t)
ax_A_bot.plot(t, v_I_t, 'k-', lw=1.2)
ax_A_bot.set_xlim(0, 1000)
ax_A_bot.set_ylim(0, 40)
ax_A_bot.set_yticks([0, 10, 20, 30, 40])
ax_A_bot.set_xlabel(r'$t$ (ms)', fontsize=12)
ax_A_bot.set_ylabel(r'$v_I$ (Hz)', fontsize=12)

# 清理 A 图脊背
for ax in [ax_A_top, ax_A_bot]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', labelsize=10)

# ==========================================
# 图 B: 相平面轨迹 (Phase Plane Trajectory)
# ==========================================
ax_B = fig.add_subplot(gs[:, 1])

# 绘制轨迹
ax_B.plot(v_E_t, v_I_t, 'k-', lw=1.0)
# 绘制初始点 (空心圆)
ax_B.plot(v0[0], v0[1], marker='o', markersize=6, markerfacecolor='white', markeredgecolor='black', zorder=5)

# 绘制零倾线
v_E_null = np.linspace(0, 60, 400)
v_I_E_null = 0.25 * v_E_null + 10  # dvE/dt = 0
v_I_I_null = v_E_null - 10         # dvI/dt = 0

ax_B.plot(v_E_null, v_I_E_null, 'k-', lw=1.2)
ax_B.plot(v_E_null, v_I_I_null, 'k-', lw=1.2)

# 添加零倾线标注文字与箭头
ax_B.annotate(r'$dv_E/dt = 0$', xy=(50, 22.5), xytext=(45, 17),
              arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
              fontsize=12, ha='left', va='center')

ax_B.text(42, 27, r'$\leftarrow dv_I/dt = 0$', fontsize=12, va='center')

# 样式美化
ax_B.set_xlim(0, 60)
ax_B.set_ylim(0, 32)
ax_B.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax_B.set_xlabel(r'$v_E$ (Hz)', fontsize=12)
ax_B.set_ylabel(r'$v_I$ (Hz)', fontsize=12)
ax_B.spines['top'].set_visible(False)
ax_B.spines['right'].set_visible(False)
ax_B.tick_params(direction='in', labelsize=10)
ax_B.text(-8, 34, 'B', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()