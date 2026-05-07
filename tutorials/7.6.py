import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
plt.style.use('./tutorial.mplstyle')

# --- 1. 定义模型函数 (与之前一致) ---
def f_u(s, xi, g, gamma, sigma=15.0):
    return np.exp(-((s - xi)**2 + (g - gamma)**2) / (2 * sigma**2))

def weight(xi, gamma, sigma_w=20.0):
    return np.exp(-((xi + gamma)**2) / (2 * sigma_w**2))

def v_inf(s, g, xi_grid, gamma_grid):
    d_xi = xi_grid[1] - xi_grid[0]
    d_gamma = gamma_grid[1] - gamma_grid[0]
    XI, GAMMA = np.meshgrid(xi_grid, gamma_grid)
    W = weight(XI, GAMMA)
    U = f_u(s, XI, g, GAMMA)
    return np.sum(W * U) * d_xi * d_gamma

# --- 2. 初始化计算空间 ---
s_values = np.linspace(-60, 60, 100) 
xi_grid = np.linspace(-90, 90, 60)   
gamma_grid = np.linspace(-90, 90, 60) 

# --- 3. 定义绘图包装函数 ---
def plot_coordinate_transform(g):
    """根据传入的注视角度 g，计算并绘制响应曲线"""
    response_curve = [v_inf(s, g, xi_grid, gamma_grid) for s in s_values]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_values, response_curve, lw=2, color='royalblue')
    
    # 动态标记峰值位置
    ax.axvline(x=-g, color='red', linestyle='--', alpha=0.8, label=f'Peak ($s = {-g}$)')
    
    ax.set_xlim(-60, 60)
    # 锁定 y 轴范围，防止拖动时画面上下跳动
    ax.set_ylim(0, 1200) 
    ax.set_xlabel('Stimulus Location on Retina $s$ (degrees)')
    ax.set_ylabel('Output Firing Rate $v_\infty$')
    ax.set_title('Coordinate Transformation: Body-centered response')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    
    plt.show()

# --- 4. 使用 interact 激活交互 ---
# 这里会自动生成一个原生的 Jupyter 滑动条，并且完美支持 VS Code
widgets.interact(
    plot_coordinate_transform, 
    g=widgets.FloatSlider(value=0, min=-30, max=30, step=2.0, description='Gaze Angle $g$')
)