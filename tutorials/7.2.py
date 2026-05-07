import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time as pytime
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1. 基础参数设置 
# ==========================================
tau_m = 10.0      # 膜时间常数 (ms) -> 决定了低通滤波的截止频率
E_L = -65.0       # 静息电位 (mV)
V_th = -50.0      # 动作电位阈值 (mV)
V_reset = -65.0   # 重置电位 (mV)
R_m = 1.0         # 膜电阻 (M ohm)

I_1 = 7.0         # 正弦振幅
I_0_high = 23.0   # 左侧：持续放电 (波谷总线电流远高于阈值)
I_0_low = 13.0    # 右侧：部分放电/整流 (波谷时电流不足以诱发动作电位)

sigma_noise = 4.5 # 提升噪声标准差，彻底打破神经元与正弦波的绝对相位锁定

# 【关键修改】减小时间步长，增加 trial 数量以支持微小的高频 bin_size
dt = 0.1          # 全局时间步长缩小为 0.1 ms，大幅提升高频下的数值精度
T = 1000.0        # 总仿真时间 (ms) 保持1000ms以获得充足的统计样本
time = np.arange(0, T, dt)
n_steps = len(time)

n_trials = 2000   # 增加试次，确保在 0.1ms 步长下的统计仍然平滑
f_list = [1.0, 50.0, 100.0]

# ==========================================
# 2. 向量化神经元仿真函数
# ==========================================
def simulate_lif(I_0, f):
    t_sec = time / 1000.0
    I_inj_array = I_0 + I_1 * np.sin(2 * np.pi * f * t_sec)
    
    v = np.full(n_trials, E_L) 
    spike_counts = np.zeros(n_steps)
    v_example = np.zeros(n_steps)
    
    for i in range(n_steps):
        noise = np.random.randn(n_trials) * sigma_noise
        I_tot = I_inj_array[i] + noise
        
        # Euler 法求解 LIF 微分方程
        dv = (E_L - v + R_m * I_tot) / tau_m * dt
        v += dv
        
        spiked = v >= V_th
        spike_counts[i] = np.sum(spiked)
        v[spiked] = V_reset
        
        # 记录第一个 trial 的电压轨迹作为画图示例
        if spiked[0]:
            v_example[i] = -20.0  # 用较高电位(-20mV)示意脉冲，方便图中观测
        else:
            v_example[i] = v[0]
            
    # 计算每个 dt 上的瞬时发放率
    rate_dt = (spike_counts / n_trials) / (dt / 1000.0)
    return v_example, rate_dt, spike_counts

# ==========================================
# 3. 运行仿真并动态绘图 (紧凑版)
# ==========================================
print(f"正在并行模拟 {len(f_list) * 2} 组实验，总计 {n_trials * len(f_list) * 2} 个 Trial...")
print(f"由于 dt={dt}ms, n_trials={n_trials}，预计耗时可能在 5-15 秒左右，请稍候...")
start_time = pytime.time()

# 【排版修改 1】缩小画布尺寸，(12, 8) 或 (10, 7) 都很适合 Notebook 屏幕浏览
fig, axes = plt.subplots(4, 2, figsize=(12, 8))

for row_idx, f in enumerate(f_list):
    v_h, rate_dt_h, counts_h = simulate_lif(I_0_high, f)
    v_l, rate_dt_l, counts_l = simulate_lif(I_0_low, f)
    
    # 动态分配统计 Bin 大小和平滑窗口
    if f == 1.0:
        show_time = 1000.0   
        bin_size = 20.0      
        smooth_ms = 20.0     
    elif f == 50.0:
        show_time = 100.0    
        bin_size = 1.0       
        smooth_ms = 1.0      
    else: 
        show_time = 50.0     
        bin_size = 0.5       
        smooth_ms = 0.5      
        
    steps_per_bin = int(bin_size / dt)
    bin_centers = np.arange(bin_size/2, T, bin_size)

    # ----------------------------------------------------
    # 第 1 行: 电压示意图
    # ----------------------------------------------------
    if f == 1.0:
        axes[0, 0].plot(time, v_h, color='black', lw=0.8)
        # 【排版修改 2】适度缩小字体 (fontsize 14 -> 11/12)
        axes[0, 0].set_title('Continuous Firing (High $I_0$)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('V (mV)', fontsize=10)
        axes[0, 0].set_xlim(0, show_time)
        axes[0, 0].set_ylim(-75, -15)
        
        axes[0, 1].plot(time, v_l, color='black', lw=0.8)
        axes[0, 1].set_title('Rectified Firing (Low $I_0$)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlim(0, show_time)
        axes[0, 1].set_ylim(-75, -15)

    # ----------------------------------------------------
    # 第 2, 3, 4 行: PSTH 与平滑率
    # ----------------------------------------------------
    plot_row = row_idx + 1 
    
    psth_h = (counts_h.reshape(-1, steps_per_bin).sum(axis=1) / n_trials) / (bin_size / 1000.0)
    psth_l = (counts_l.reshape(-1, steps_per_bin).sum(axis=1) / n_trials) / (bin_size / 1000.0)
    
    rate_smooth_h = gaussian_filter1d(rate_dt_h, sigma=smooth_ms / dt)
    rate_smooth_l = gaussian_filter1d(rate_dt_l, sigma=smooth_ms / dt)
    
    max_y = max(rate_smooth_h.max(), rate_smooth_l.max()) * 1.3
    if max_y < 10.0: max_y = 10.0  

    # 左侧：连续放电
    axes[plot_row, 0].bar(bin_centers, psth_h, width=bin_size*0.9, color='lightgray', align='center')
    axes[plot_row, 0].plot(time, rate_smooth_h, color='steelblue', lw=2.0) # 线宽稍微调细
    axes[plot_row, 0].set_ylabel(f'Rate (Hz)\n[f = {int(f)} Hz]', fontsize=10, fontweight='bold')
    axes[plot_row, 0].set_xlim(0, show_time)
    axes[plot_row, 0].set_ylim(0, max_y)
    
    # 右侧：带截断整流
    axes[plot_row, 1].bar(bin_centers, psth_l, width=bin_size*0.9, color='lightgray', align='center')
    axes[plot_row, 1].plot(time, rate_smooth_l, color='tomato', lw=2.0)
    axes[plot_row, 1].set_xlim(0, show_time)
    axes[plot_row, 1].set_ylim(0, max_y)

axes[3, 0].set_xlabel('Time (ms)', fontsize=11)
axes[3, 1].set_xlabel('Time (ms)', fontsize=11)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)
    # 【排版修改 3】调小刻度字体
    ax.tick_params(axis='both', which='major', labelsize=9)
    
# 【排版修改 4】使用 tight_layout 自动收缩边距，代替手动 adjust
plt.tight_layout()
print(f"计算完成，耗时: {pytime.time() - start_time:.2f} 秒。")
plt.show()