import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import odeint
plt.style.use('./tutorial.mplstyle')

# --- 1. 参数设置 ---
M_EE = 1.25
M_IE = 1.0
M_II = 0.0
M_EI = -1.0
gamma_E = -10.0
gamma_I = 10.0
tau_E = 10.0
tau_I = 50.0  # Hopf 分岔后的参数

# --- 2. 定义动力学方程 ---
def firing_rate_model(v, t):
    v_E, v_I = v
    input_E = M_EE * v_E + M_EI * v_I - gamma_E
    input_I = M_II * v_I + M_IE * v_E - gamma_I
    rate_E = max(input_E, 0)
    rate_I = max(input_I, 0)
    dvE_dt = (-v_E + rate_E) / tau_E
    dvI_dt = (-v_I + rate_I) / tau_I
    return [dvE_dt, dvI_dt]

# --- 3. 运行数值模拟 ---
t = np.linspace(0, 1000, 5000) # 增加采样率以精确捕捉转折点
v0 = [40.0, 22.0]
solution = odeint(firing_rate_model, v0, t)
v_E_t = solution[:, 0]
v_I_t = solution[:, 1]

# 提取稳定后的极限环数据 (抛弃前 500ms 的瞬态)
steady_mask = t > 500
v_E_steady = v_E_t[steady_mask]
v_I_steady = v_I_t[steady_mask]

# 寻找 dv_E/dt = 0 的精确转折点 (即 v_E 的局部最小值)
min_idx = np.argmin(v_E_steady)
v_E_turn = v_E_steady[min_idx]
v_I_turn = v_I_steady[min_idx]

# --- 4. 绘图 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# 扩展画布宽度以容纳第三个子图
fig = plt.figure(figsize=(11.5, 4.2), layout='constrained')
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)

# ==========================================
# 图 A: 随时间演化的放电率
# ==========================================
ax_A_top = fig.add_subplot(gs[0, 0])
ax_A_bot = fig.add_subplot(gs[1, 0])

ax_A_top.plot(t, v_E_t, 'k-', lw=1.2)
ax_A_top.set_xlim(0, 1000)
ax_A_top.set_ylim(0, 60)
ax_A_top.set_yticks([0, 20, 40, 60])
ax_A_top.set_ylabel(r'$v_E$ (Hz)', fontsize=12)
ax_A_top.text(-250, 65, 'A', fontsize=14, fontweight='bold', va='center')

ax_A_bot.plot(t, v_I_t, 'k-', lw=1.2)
ax_A_bot.set_xlim(0, 1000)
ax_A_bot.set_ylim(0, 40)
ax_A_bot.set_yticks([0, 10, 20, 30, 40])
ax_A_bot.set_xlabel(r'$t$ (ms)', fontsize=12)
ax_A_bot.set_ylabel(r'$v_I$ (Hz)', fontsize=12)

for ax in [ax_A_top, ax_A_bot]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', labelsize=10)

# ==========================================
# 图 B: 全局相平面轨迹
# ==========================================
ax_B = fig.add_subplot(gs[:, 1])

ax_B.plot(v_E_t, v_I_t, 'k-', lw=1.2)
ax_B.plot(v0[0], v0[1], marker='o', markersize=6, markerfacecolor='white', markeredgecolor='black', zorder=5)

v_E_null = np.linspace(0, 60, 400)
ax_B.plot(v_E_null, 0.25 * v_E_null + 10, 'k-', lw=1.0)  # v_E nullcline
ax_B.plot(v_E_null, v_E_null - 10, 'k-', lw=1.0)         # v_I nullcline

# 标记 C 图的放大区域
rect = patches.Rectangle((-1, 9.5), 5, 2.5, fill=False, edgecolor='gray', linestyle='--', lw=1)
ax_B.add_patch(rect)
ax_B.text(4, 12.2, 'Zoom (Fig C)', color='gray', fontsize=10)

ax_B.set_xlim(-2, 60)
ax_B.set_ylim(0, 32)
ax_B.set_xlabel(r'$v_E$ (Hz)', fontsize=12)
ax_B.set_ylabel(r'$v_I$ (Hz)', fontsize=12)
ax_B.spines['top'].set_visible(False)
ax_B.spines['right'].set_visible(False)
ax_B.tick_params(direction='in', labelsize=10)
ax_B.text(-12, 34, 'B', fontsize=14, fontweight='bold')

# ==========================================
# 图 C: 截距邻域放大图 (The Microscopic View)
# ==========================================
ax_C = fig.add_subplot(gs[:, 2])

# 绘制各种几何边界
v_E_zoom = np.linspace(-1, 5, 100)
# 1. 真实的零倾线 (dv_E/dt = 0)
ax_C.plot(v_E_zoom, 0.25 * v_E_zoom + 10, 'k-', lw=1.5, label='Nullcline ($dv_E/dt=0$)')
# 2. 激活线 (Switch Line, 中括号内为0)
ax_C.plot(v_E_zoom, 1.25 * v_E_zoom + 10, color='gray', linestyle='--', lw=1.5, label='Activation Line')
# 3. y轴边界线 (v_E = 0)
ax_C.axvline(0, color='black', lw=0.8, alpha=0.5)

# 绘制极限环轨迹
ax_C.plot(v_E_steady, v_I_steady, 'b-', lw=1.8, label='Limit Cycle')

# 标记 (0,10) 截距点
ax_C.plot(0, 10, 'kx', markersize=8, markeredgewidth=2, label='Intercept (0,10)')

# 标记转折点
ax_C.plot(v_E_turn, v_I_turn, marker='o', markersize=6, markerfacecolor='white', markeredgecolor='blue', zorder=5)
ax_C.annotate(f'Turn Point\n({v_E_turn:.2f}, {v_I_turn:.2f})', 
              xy=(v_E_turn, v_I_turn), xytext=(v_E_turn + 0.5, v_I_turn - 0.4),
              arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=10)

ax_C.set_xlim(-0.5, 3.5)
ax_C.set_ylim(9.5, 12)
ax_C.set_xlabel(r'$v_E$ (Hz)', fontsize=12)
ax_C.set_ylabel(r'$v_I$ (Hz)', fontsize=12)
ax_C.spines['top'].set_visible(False)
ax_C.spines['right'].set_visible(False)
ax_C.tick_params(direction='in', labelsize=10)
ax_C.legend(fontsize=9, loc='upper right', frameon=False)
ax_C.text(-1, 12.2, 'C', fontsize=14, fontweight='bold')

plt.show()