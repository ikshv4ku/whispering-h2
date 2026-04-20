"""
whispering-h2 — Attention Visualisation Notebook (Lab A7)

This notebook demonstrates multi-head self-attention visualisation,
directly mirroring Lab A7: "Transformer Embedding + Self-Attention".

Key questions answered:
  Q1. Which attention heads attend to the planet feature prefix tokens?
  Q2. Do any heads behave as 'previous-token heads' (attend to t-1)?
  Q3. Do deeper layers show more semantic (myth-content-focused) attention?
  Q4. Where does the model focus when generating fire-themed myths vs
      ice-themed myths?

To run:
  cd /home/pradyuman/whispering-h2
  jupyter notebook notebooks/attention_viz.ipynb
"""

import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader import load_config, load_processed, serialise_planet_features
from src.transformer_model import build_tokeniser, load_checkpoint, build_feature_vector
from src.evaluate import plot_attention_heatmap

cfg = load_config()
PROJECT_ROOT = Path().resolve().parent if Path().resolve().name == "notebooks" else Path().resolve()

# ─── Cell 1: Load model ──────────────────────────────────────────────────────
ckpt_path = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"] / "best_model.pt"
if not ckpt_path.exists():
    print("No checkpoint found. Run `python src/train.py --smoke_test` first.")
else:
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokeniser = build_tokeniser(cfg)
    model     = load_checkpoint(str(ckpt_path), cfg, tokeniser)
    model.eval()
    print(f"Model loaded. Device: {device}")

    test_df, test_pairs, _ = load_processed("test", cfg)
    planet_lookup = {r["planet_name"]: r for _, r in test_df.iterrows()
                     if "planet_name" in test_df.columns}

    # Pick two contrasting planets: one fire, one ice
    fire_pair = next((p for p in test_pairs if p.get("archetype") == "fire"), test_pairs[0])
    ice_pair  = next((p for p in test_pairs if p.get("archetype") == "ice"),  test_pairs[1])

    # ─── Cell 2: Extract attention weights for fire planet ────────────────────
    fire_input = f"<myth> <fire> {fire_pair['prefix']}\n{fire_pair['story'][:150]}"
    tokens_f, attn_f = model.get_attention_weights(fire_input, device=device)
    print(f"Fire planet tokens ({len(tokens_f)}): {' '.join(tokens_f[:15])} …")

    # ─── Cell 3: Plot attention for layer 0 (fire planet) ────────────────────
    plot_attention_heatmap(
        tokens_f, attn_f, layer_idx=0,
        save_dir=PROJECT_ROOT / "outputs" / "attention",
        max_tokens=30,
    )
    print("Layer 0 attention heatmap saved (fire planet).")

    # ─── Cell 4: Plot attention for layer 5 (fire planet) ────────────────────
    n_layers = len(attn_f)
    mid_layer = min(5, n_layers - 1)
    plot_attention_heatmap(
        tokens_f, attn_f, layer_idx=mid_layer,
        save_dir=PROJECT_ROOT / "outputs" / "attention",
        max_tokens=30,
    )
    print(f"Layer {mid_layer} attention heatmap saved (fire planet).")

    # ─── Cell 5: Ice planet attention ────────────────────────────────────────
    ice_input = f"<myth> <ice> {ice_pair['prefix']}\n{ice_pair['story'][:150]}"
    tokens_i, attn_i = model.get_attention_weights(ice_input, device=device)

    plot_attention_heatmap(
        tokens_i, attn_i, layer_idx=0,
        save_dir=PROJECT_ROOT / "outputs" / "attention",
        max_tokens=30,
    )
    print("Layer 0 attention heatmap saved (ice planet).")

    # ─── Cell 6: Head entropy analysis ───────────────────────────────────────
    # Compute attention entropy for each head in layer 0:
    # H(head) = -Σ p_ij log p_ij  averaged over all query positions
    # Low entropy → focused (attends to 1-2 tokens);  High → diffuse
    print("\n── Head Attention Entropy (Layer 0, Fire Planet) ──")
    print("  Head  |  Entropy (lower = more focused)")
    attn_l0 = attn_f[0]  # (n_heads, T, T)
    for h in range(attn_l0.shape[0]):
        head_mat = attn_l0[h].numpy()  # (T, T)
        # Avoid log(0)
        ent_per_row = -np.sum(head_mat * np.log(head_mat + 1e-12), axis=-1)
        avg_ent = float(np.mean(ent_per_row))
        bar = "█" * int(avg_ent * 3)
        print(f"  Head {h:>2} | {avg_ent:5.3f}  {bar}")

    # ─── Cell 7: Prefix attention fraction ───────────────────────────────────
    # For each head, compute what fraction of attention is on prefix tokens
    print("\n── Fraction of Attention on Feature Prefix (Layer 0, Fire Planet) ──")
    prefix_boundary = next(
        (i for i, t in enumerate(tokens_f) if "PLANET" in t or "planet" in t.lower()), len(tokens_f)//3
    )
    print(f"  (Prefix boundary at token {prefix_boundary}: '{tokens_f[prefix_boundary]}')")
    for h in range(attn_l0.shape[0]):
        head_mat = attn_l0[h, prefix_boundary:, :prefix_boundary].numpy()
        prefix_frac = float(head_mat.mean())
        bar = "█" * int(prefix_frac * 40)
        print(f"  Head {h:>2} | {prefix_frac:.4f}  {bar}")

    print("\nAttention visualisation complete. Plots saved to outputs/attention/")

print("\nNote: Run `python src/train.py --smoke_test` first to create a checkpoint.")
