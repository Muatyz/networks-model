import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 生物学参数设置 (已修正动力学平衡)
# ==========================================
tau_E = 6.7   # Mitral 细胞时间常数 (ms)
tau_I = 6.7   # Granule 细胞时间常数 (ms)
tau_syn = 3.0 # 突触延迟 (ms) - 突破二维 Bendixson 准则的关键

# 突触连接权重 
w_EE = 0.0    # 严格遵循解剖学
w_II = 0.0    # 严格遵循解剖学
w_EI = 5.0    # 降低抑制权重，设定最大抑制电流上限
w_IE = 12.0   # 保持足够的兴奋驱动以激活 Granule

# 激活函数参数
# 适当降低增益 a，恢复平滑的谐波振荡而非弛豫振荡
a_E, theta_E = 2.5, 2.5  
a_I, theta_I = 2.5, 6.0  # 匹配新的 w_IE 权重，使平衡点落在最敏感区

def sigmoid(x, a, theta):
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))

# ==========================================
# 2. 嗅探周期输入 h_E(t)
# ==========================================
def h_E_func(t):
    h_base = 0.5   
    h_amp = 4.5    # 峰值总输入为 h_base + h_amp = 5.0
    t_peak = 200.0 
    sigma = 40.0   
    return h_base + h_amp * np.exp(-0.5 * ((t - t_peak) / sigma)**2)

# ==========================================
# 3. 4D 网络动力系统微分方程
# ==========================================
def ei_network_4d(state, t):
    v_E, s_E, v_I, s_I = state # v: 细胞放电率, s: 突触后效(延迟变量)
    
    h_E_t = h_E_func(t)
    
    # 细胞膜电位驱动放电率变化 (没有 w_EE 和 w_II)
    dvE = (-v_E + sigmoid(h_E_t - w_EI * s_I, a_E, theta_E)) / tau_E
    dvI = (-v_I + sigmoid(w_IE * s_E, a_I, theta_I)) / tau_I
    
    # 突触后电位的低通滤波生成 (引入生物学延迟)
    dsE = (-s_E + v_E) / tau_syn
    dsI = (-s_I + v_I) / tau_syn
    
    return [dvE, dsE, dvI, dsI]

# ==========================================
# 4. 求解与数据准备
# ==========================================
t_span = np.linspace(0, 400, 1500) # 模拟 400ms 嗅探周期
y0 = [0.1, 0.1, 0.1, 0.1]          # 初始静息态
sol = odeint(ei_network_4d, y0, t_span)

v_E_t = sol[:, 0] # Mitral 放电率
v_I_t = sol[:, 2] # Granule 放电率
s_I_t = sol[:, 3] # Granule 对 Mitral 的实际抑制电流

# 构建符合原书图解的 Mitral "表观膜电位"
# = 正向嗅探慢波驱动 (h_E) - 高频突触抑制涟漪 (w_EI * s_I)
membrane_potential_E = h_E_func(t_span) - w_EI * s_I_t

# ==========================================
# 5. 绘制动画
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：时域上的膜电位 (大Bump + 高频涟漪) ---
ax1.set_xlim(0, 400)
ax1.set_ylim(-1, 10)
ax1.set_title("Mitral Cell Potential during Sniff Cycle")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Membrane Potential")
line_uE, = ax1.plot([], [], 'b-', lw=1.5, label='Mitral Output (Bump + Oscillation)')
line_hE, = ax1.plot(t_span, h_E_func(t_span), 'k--', lw=1.5, alpha=0.3, label='Sniff Input $h_E(t)$')
ax1.legend(loc='upper right')

# --- 右图：相空间轨迹 ---
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 1.0)
ax2.set_title("Phase Space: Mitral vs Granule Firing Rate")
ax2.set_xlabel("Mitral Firing Rate ($v_E$)")
ax2.set_ylabel("Granule Firing Rate ($v_I$)")
line_phase, = ax2.plot([], [], 'r-', lw=1.2, alpha=0.6)
point_phase, = ax2.plot([], [], 'ro', markersize=8)

def init():
    line_uE.set_data([], [])
    line_phase.set_data([], [])
    point_phase.set_data([], [])
    return line_uE, line_phase, point_phase

def animate(i):
    # 每帧跳跃渲染，加速动画播放
    step = 15 
    idx = min(i * step, len(t_span) - 1)
    
    line_uE.set_data(t_span[:idx], membrane_potential_E[:idx])
    line_phase.set_data(v_E_t[:idx], v_I_t[:idx])
    point_phase.set_data([v_E_t[idx]], [v_I_t[idx]]) # 修复点数据格式
    
    return line_uE, line_phase, point_phase

frames = len(t_span) // 15
anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=40, blit=True)

# 防止静态图重复显示
plt.close()

# 在 Jupyter 中输出带控制条的 HTML5 视频
display(HTML(anim.to_jshtml()))