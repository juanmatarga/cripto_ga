"""Generate clean evolution progress figure for the paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = 'results/experiment_seed123_20260307_193201/evolution_log.json'
OUT_PATH = 'paper/figures/fig3_evolution.png'

with open(LOG_PATH) as f:
    log = json.load(f)

# Aggregate across islands per generation
gens_best = []
gens_mean = []

for entry in log:
    island_bests = []
    island_means = []
    for k, v in entry.items():
        if isinstance(v, dict) and 'best_fitness' in v:
            bf = v['best_fitness']
            mf = v.get('mean_fitness', 0)
            if bf > -900:
                island_bests.append(bf)
            if mf > -900:
                island_means.append(mf)
    gens_best.append(max(island_bests) if island_bests else 0)
    gens_mean.append(np.mean(island_means) if island_means else 0)

gens = np.arange(len(log))
best = np.array(gens_best)
mean = np.array(gens_mean)
cum_best = np.maximum.accumulate(best)

# Smooth mean with rolling window
window = 5
mean_smooth = np.convolve(mean, np.ones(window)/window, mode='same')
# Fix edges
mean_smooth[:window//2] = mean[:window//2]
mean_smooth[-window//2:] = mean[-window//2:]

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

# Shaded band: min valid to best per generation
# Use mean - std and best as the band
ax.fill_between(gens, np.clip(mean_smooth - 1.5, 0, None), best,
                alpha=0.12, color='#2196F3', label='Rango de la población')

# Scatter of raw mean values (faint)
ax.scatter(gens, mean, alpha=0.15, s=12, color='#2196F3', zorder=2)

# Smoothed mean line
ax.plot(gens, mean_smooth, color='#1565C0', linewidth=2.5,
        label='Aptitud promedio', zorder=3)

# Cumulative best (staircase)
ax.plot(gens, cum_best, color='#1565C0', linewidth=2, linestyle='--',
        label='Mejor encontrada', zorder=4)

ax.set_xlabel('Generación', fontsize=13)
ax.set_ylabel('Aptitud (fitness compuesto)', fontsize=13)
ax.set_title('Progresión de la aptitud durante la evolución', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim(0, len(log)-1)
ax.set_ylim(0, max(cum_best) * 1.08)
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
print(f"Generations: {len(log)}")
print(f"Best: {cum_best[0]:.2f} -> {cum_best[-1]:.2f}")
print(f"Mean: {mean[0]:.2f} -> {mean[-1]:.2f}")
