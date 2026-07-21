"""
Generate Figure 2: RACF Three-Stage Mechanism Flowchart (FIXED)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

dst = os.path.dirname(os.path.abspath(__file__))  # save to current dir

plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('Figure 2: RACF Three-Stage Coalition Formation Mechanism', fontsize=14, fontweight='bold', pad=15)

stages = [
    (1, 8, 10, 2.5, 'Stage I: Reputation-Filtered\nInitial Coalition Formation\n\nSuppliers greedily join\nreputation-feasible coalitions\nmaximizing individual utility', '#4472C4'),
    (1, 4.5, 10, 2.5, 'Stage II: TOPSIS-Based\nPriority Reallocation\n\nMulti-criteria ranking repairs\ninfeasible coalitions via\ntargeted supplier transfers', '#ED7D31'),
    (1, 1, 10, 2.5, 'Stage III: Stability-Driven\nRefinement\n\nSwitching and splitting\noperations converge to\nfeasible-switch-stable partition', '#70AD47'),
]
for x, y, w, h, text, color in stages:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Arrows between stages
for y1, y2 in [(7.8, 7.2), (4.3, 3.7)]:
    ax.annotate('', xy=(6, y1), xytext=(6, y2), arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(6.3, (y1+y2)/2, 'Infeasible\ncoalitions', ha='left', fontsize=9, style='italic')

# Right side annotations
ax.text(11.2, 9.2, 'Output:\nFeasibility flags', ha='center', fontsize=9, style='italic')
ax.text(11.2, 5.7, 'Output:\nRepaired partition', ha='center', fontsize=9, style='italic')
ax.text(11.2, 2.2, r'Output: feasible-switch-stable' + '\n' + r'partition $\Pi^\star$', ha='center', fontsize=9, style='italic')

plt.tight_layout()
fig.savefig(os.path.join(dst, 'fig2_racf_flowchart.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 2 regenerated: Nash-stable → feasible-switch-stable')
