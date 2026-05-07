import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./tutorial.mplstyle')

# ==========================================
# 1-4. 数据生成与计算部分保持不变
# ==========================================
theta = np.linspace(-180, 180, 200) 
theta_rad = np.radians(theta)       
np.random.seed(42) 
base_cosine = 0.8 * np.cos(theta_rad) 
noise = 1.5 * np.random.randn(200)
h_input = base_cosine + noise

lambda_1 = 0.9
gain_mu1 = 1 / (1 - lambda_1)  
gain_other = 1.0               

mu_values = np.arange(10)
h_amplitudes = np.zeros(10)
v_amplitudes = np.zeros(10)
v_inf = np.zeros_like(theta_rad)

for mu in mu_values:
    if mu == 0:
        h_cos_int = np.trapezoid(h_input, theta_rad) / (2 * np.pi)
        h_sin_int = 0.0
    else:
        h_cos_int = np.trapezoid(h_input * np.cos(mu * theta_rad), theta_rad) / np.pi
        h_sin_int = np.trapezoid(h_input * np.sin(mu * theta_rad), theta_rad) / np.pi
    
    h_amplitudes[mu] = np.sqrt(h_cos_int**2 + h_sin_int**2)
    current_gain = gain_mu1 if mu == 1 else gain_other
    v_amplitudes[mu] = h_amplitudes[mu] * current_gain
    
    if mu == 0:
        v_inf += current_gain * h_cos_int
    else:
        v_inf += current_gain * (h_cos_int * np.cos(mu * theta_rad) + h_sin_int * np.sin(mu * theta_rad))

# ==========================================
# 5. 调整后的绘图设置 (更适合 Notebook)
# ==========================================
# 减小 figsize：例如 (10, 6) 或者 (8, 6)
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# 统一调整字体大小
title_fs = 12
label_fs = 10
tick_fs = 9

# Panel A
axs[0, 0].plot(theta, h_input, color='black', linewidth=1.2)
axs[0, 0].set_title('A. Input $h(\\theta)$', loc='left', fontsize=title_fs, fontweight='bold')
axs[0, 0].set_ylabel('$h$', fontsize=label_fs, rotation=0, labelpad=10)
axs[0, 0].set_xlabel('$\\theta$ (deg)', fontsize=label_fs)
axs[0, 0].set_xticks([-180, -90, 0, 90, 180])
axs[0, 0].tick_params(labelsize=tick_fs)
axs[0, 0].grid(True, linestyle='--', alpha=0.5)

# Panel B
axs[0, 1].plot(theta, v_inf, color='black', linewidth=1.2)
axs[0, 1].set_title('B. Network Output $v_\\infty(\\theta)$', loc='left', fontsize=title_fs, fontweight='bold')
axs[0, 1].set_ylabel('$v$', fontsize=label_fs, rotation=0, labelpad=10)
axs[0, 1].set_xlabel('$\\theta$ (deg)', fontsize=label_fs)
axs[0, 1].set_xticks([-180, -90, 0, 90, 180])
axs[0, 1].tick_params(labelsize=tick_fs)
axs[0, 1].grid(True, linestyle='--', alpha=0.5)

max_h_amp = np.max(h_amplitudes) * 1.15
max_v_amp = np.max(v_amplitudes) * 1.15

# Panel C
axs[1, 0].bar(mu_values, h_amplitudes, color='black', width=0.8)
axs[1, 0].set_title('C. Fourier Amplitudes of Input', loc='left', fontsize=title_fs, fontweight='bold')
axs[1, 0].set_ylabel('Amplitude', fontsize=label_fs)
axs[1, 0].set_xlabel('Spatial Frequency $\\mu$', fontsize=label_fs)
axs[1, 0].set_xticks(mu_values)
axs[1, 0].set_ylim(0, max_h_amp)
axs[1, 0].tick_params(labelsize=tick_fs)
axs[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

# Panel D
axs[1, 1].bar(mu_values, v_amplitudes, color='black', width=0.8)
axs[1, 1].set_title('D. Fourier Amplitudes of Output', loc='left', fontsize=title_fs, fontweight='bold')
axs[1, 1].set_ylabel('Amplitude', fontsize=label_fs)
axs[1, 1].set_xlabel('Spatial Frequency $\\mu$', fontsize=label_fs)
axs[1, 1].set_xticks(mu_values)
axs[1, 1].set_ylim(0, max_v_amp)
axs[1, 1].tick_params(labelsize=tick_fs)
axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

# 调整子图间距
plt.tight_layout()
plt.show()