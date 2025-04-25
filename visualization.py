import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

results_file = 'C:/Users/DELL/Desktop/project/results.xlsx'
score = pd.read_excel(results_file, sheet_name='Result_all')
structure_score = score['structure']
function_score = score['function']

## Define the range of x and y and create a grid
x = y = np.linspace(-120, 120, 400)
X, Y = np.meshgrid(x, y)
F = np.where(X + Y != 0, (2 * X * Y) / (X + Y), np.nan)

## Set colors and fonts
colors = ['#1D5E22', '#0B7F14', '#64C06C', '#B4DAB7', '#C4C3C1', '#ff7f0e', '#FFEDCB', '#055BA8', '#61A5C2']
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})
## Create graphics
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['BL5A20', 'BL5A15', 'BL5A10', 'BL5A5', 'BL15', 'NatureWangXia', 'Organoid', 'Stomach', 'Liver', 'Target']
## N-1 rows of data before drawing
for i in range(len(structure_score) - 1):
    ax.scatter(structure_score[i], function_score[i], color=colors[i], alpha=1, marker='^', s=400, edgecolors='black', label=labels[i], zorder=3)
## Draw the last line of data as a five-pointed star
ax.scatter(structure_score[9], function_score[9], color='red', alpha=1, marker='*', s=500, edgecolors='black', label=labels[9], zorder=3)

## Draw isoline
F_masked = np.where((X < 0) | (Y < 0), np.nan, F)
contour = ax.contour(X, Y, F_masked, levels=[20, 40, 60, 80, 100], linestyles='dashed', colors='#7F7F7F', zorder=1, linewidths=1)

## Set coordinate axes and add graphic borders.
ax.axhline(0, color='black', lw=1.5)
ax.axvline(0, color='black', lw=1.5)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_xlabel('Structure Score', labelpad=100, fontsize=16, color='black')
ax.set_ylabel('Function Score', labelpad=90, fontsize=16, color='black')
ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False, handleheight=2.5, fontsize=12)
ax.patch.set(edgecolor='black', linewidth='1.5')
## Set the drawing range, scale and scale
ax.set(xlim=(-60, 110), ylim=(-60, 110), aspect='equal', xticks=[-50, 50, 100], yticks=[-50, 0, 50, 100])
ax.tick_params(axis='both', labelsize=13)

## Adjust layout and display graphics
plt.subplots_adjust(right=0.85, left=0.05, bottom=0.15, top=0.9)
## Save the resized picture.
plt.savefig('results/Figs/score.png', dpi=500)