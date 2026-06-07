"""
CN Attack — RIR gradient figure
Output: rir_attack.png + rir_attack.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============ Data (3-seed averages) ============
labels = ['Clean\n(baseline)', 'Natural shift\n(VMMRdb)',
          'Random noise\n(eps=8/255)', 'PGD-40\n(eps=8/255)']
rir    = [1.334, 0.982, 0.981, 0.025]
err    = [0.14,  0.052, 0.031, 0.002]

colors = ['#1D9E75', '#BA7517', '#BA7517', '#E24B4A']

# ============ Plot ============
fig, ax = plt.subplots(figsize=(7.5, 5))

bars = ax.bar(labels, rir, yerr=err, color=colors,
              width=0.6, capsize=5, edgecolor='none',
              error_kw={'elinewidth': 1.5, 'ecolor': '#5F5E5A'})

# Threshold line at RIR = 1
ax.axhline(y=1.0, color='#A32D2D', linestyle='--', linewidth=1.5, zorder=0)
# Label just above the dashed line, right side
ax.text(3.45, 1.075, 'RIR = 1 threshold',
        color='#A32D2D', ha='right', va='center', fontsize=10)
ax.text(3.45, 1.025, '(immunization fails below)',
        color='#A32D2D', ha='right', va='center', fontsize=9)

# Value labels on bars
for bar, v, e in zip(bars, rir, err):
    ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.03,
            f'{v:.3f}', ha='center', va='bottom',
            fontsize=11, fontweight='medium')

ax.set_ylabel('RIR  (higher = stronger immunization)', fontsize=12)
ax.set_ylim(0, 1.65)
ax.set_yticks(np.arange(0, 1.7, 0.2))
ax.set_title('CN immunization collapses under attack\n(ResNet18 + Stanford Cars, 3 seeds)',
             fontsize=13, pad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('rir_attack.png', dpi=200, bbox_inches='tight')
plt.savefig('rir_attack.pdf', bbox_inches='tight')
print("Saved: rir_attack.png and rir_attack.pdf")
