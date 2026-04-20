"""
whispering-h2 — Exploration Notebook

This notebook provides:
  1. Dataset statistics for both the exoplanet and mythology corpora
  2. Feature distribution plots (temperature, radius, mass, orbital period)
  3. Archetype class balance visualisation (fire / ice / gas / myth)
  4. Myth corpus text statistics (sentence length distribution, top bigrams)
  5. Sample planet feature serialisations (the text passed to GPT-2)

To run:
  cd /home/pradyuman/whispering-h2
  jupyter notebook notebooks/exploration.ipynb
"""

# ─── Cell 0: Setup ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 120
import seaborn as sns
sns.set_theme(style="darkgrid")

from src.data_loader import load_config, load_processed, serialise_planet_features
cfg = load_config()
print("Config loaded. Processing directory:", cfg["paths"]["processed_dir"])

# ─── Cell 1: Load data ──────────────────────────────────────────────────────
train_df, train_pairs, train_sents = load_processed("train", cfg)
test_df,  test_pairs,  test_sents  = load_processed("test",  cfg)

print(f"Train planets : {len(train_df)}")
print(f"Test  planets : {len(test_df)}")
print(f"Train sentences: {len(train_sents)}")
print(f"Train pairs   : {len(train_pairs)}")
print("\\nColumns:", list(train_df.columns[:10]))

# ─── Cell 2: Archetype Distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
counts = train_df["myth_archetype"].value_counts()
colors = {"fire": "#E63946", "ice": "#A8DADC", "gas": "#F4A261", "myth": "#457B9D"}
axes[0].pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=[colors.get(c, "#888") for c in counts.index],
    startangle=90,
)
axes[0].set_title("Planet Archetype Distribution (Train)")

# Bar chart with std-dev of key feature per archetype
arch_temp = train_df.groupby("myth_archetype")["eq_temp_K"].agg(["mean", "std"])
axes[1].bar(arch_temp.index, arch_temp["mean"],
            yerr=arch_temp["std"], capsize=5,
            color=[colors.get(c, "#888") for c in arch_temp.index])
axes[1].set_title("Mean Equilibrium Temperature by Archetype")
axes[1].set_ylabel("T_eq (K)")

plt.tight_layout()
plt.savefig("../outputs/archetype_distribution.png", bbox_inches="tight")
plt.show()
print("Archetype distribution plot saved.")

# ─── Cell 3: Feature Distributions ──────────────────────────────────────────
feature_cols = {
    "eq_temp_K":           "Equilibrium Temperature (K)",
    "radius_earth":        "Planet Radius (R_earth)",
    "mass_earth":          "Planet Mass (M_earth)",
    "orbital_period_days": "Orbital Period (days)",
    "eccentricity":        "Orbital Eccentricity",
    "distance_pc":         "Distance from Earth (pc)",
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (col, label) in zip(axes, feature_cols.items()):
    if col not in train_df.columns:
        continue
    data = train_df[col].dropna()
    # Clip extreme outliers for visualisation
    data = data.clip(data.quantile(0.01), data.quantile(0.99))
    for arch in train_df["myth_archetype"].unique():
        subset = train_df[train_df["myth_archetype"] == arch][col].dropna()
        subset = subset.clip(data.min(), data.max())
        sns.kdeplot(subset, ax=ax, label=arch, color=colors.get(arch, "#888"), fill=True, alpha=0.3)
    ax.set_xlabel(label)
    ax.set_title(label)
    ax.legend(fontsize=7)

plt.suptitle("Feature Distributions by Myth Archetype", fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/feature_distributions.png", bbox_inches="tight")
plt.show()
print("Feature distributions plot saved.")

# ─── Cell 4: Habitability Analysis ──────────────────────────────────────────
print("\\n── Habitability Statistics ──")
hab = train_df[train_df["is_habitable"]]
print(f"Habitable planets in train: {len(hab)} ({100*len(hab)/len(train_df):.1f}%)")
print(f"Mean temp (habitable):  {hab['eq_temp_K'].mean():.1f} K")
print(f"Mean radius (habitable): {hab['radius_earth'].mean():.2f} R_earth")
print(f"Mean mass (habitable):  {hab['mass_earth'].mean():.2f} M_earth")

# Scatter: radius vs temperature coloured by habitability
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    train_df["eq_temp_K"], train_df["radius_earth"],
    c=train_df["is_habitable"].astype(int),
    cmap="RdYlGn", alpha=0.6, s=20,
)
ax.set_xlabel("Equilibrium Temperature (K)")
ax.set_ylabel("Planet Radius (R_earth)")
ax.set_title("Planet Habitability (Green = Habitable)")
ax.axvspan(200, 350, alpha=0.1, color="green", label="Habitable zone T")
ax.axhline(y=2.5, color="green", linestyle="--", alpha=0.5, label="Radius threshold")
plt.colorbar(sc, label="Habitable")
ax.legend()
ax.set_xlim(0, 3000)
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig("../outputs/habitability_scatter.png", bbox_inches="tight")
plt.show()

# ─── Cell 5: Mythology Corpus Statistics ────────────────────────────────────
print("\\n── Mythology Corpus Statistics ──")
sent_lengths = [len(s.split()) for s in train_sents]
print(f"Sentences in training set : {len(train_sents)}")
print(f"Mean tokens per sentence  : {np.mean(sent_lengths):.1f}")
print(f"Median tokens per sentence: {np.median(sent_lengths):.1f}")
print(f"Max tokens                : {max(sent_lengths)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(sent_lengths, bins=40, color="#457B9D", edgecolor="white")
ax1.set_xlabel("Tokens per Sentence")
ax1.set_title("Myth Sentence Length Distribution")

# Top bigrams in mythology text
from collections import Counter
import re

all_tokens = []
for s in train_sents[:2000]:
    toks = re.findall(r"\\b[a-z]+\\b", s.lower())
    all_tokens.extend(toks)

bigrams = Counter(zip(all_tokens, all_tokens[1:]))
top_bg  = bigrams.most_common(15)
bg_labels, bg_counts = zip(*top_bg)
bg_labels = [f"{a} {b}" for (a, b) in bg_labels]
ax2.barh(bg_labels[::-1], list(bg_counts)[::-1], color="#E63946")
ax2.set_xlabel("Count")
ax2.set_title("Top Bigrams in Myth Corpus")

plt.tight_layout()
plt.savefig("../outputs/corpus_statistics.png", bbox_inches="tight")
plt.show()

# ─── Cell 6: Sample Serialised Features ─────────────────────────────────────
print("\\n── Sample Planet Feature Serialisations (GPT-2 Prefix Input) ──")
for _, row in train_df.head(5).iterrows():
    prefix = serialise_planet_features(row)
    print(f"  {row.get('planet_name', 'Unknown')[:20]:<20} →  {prefix[:90]}")
print("\\nExploration notebook complete.")
