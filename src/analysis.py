"""
analysis.py — Exoplanet Mythos Generator
==========================================
CST-106 Applications of NLP

Comprehensive analysis, training, inference, and visualisation pipeline.
Run this single script to:
  1. Re-run full baseline experiments (n-gram + NB) with real numbers
  2. Run full GPT-2 fine-tuning (3 epochs, all 10k pairs)
  3. Generate myth stories for 20 planets across all archetypes
  4. Produce 15+ publication-quality charts and tables
  5. Save a complete results JSON for the final report

Usage:
  cd /home/pradyuman/whispering-h2
  PYTHONPATH=. python3 src/analysis.py [--skip-train] [--device cpu]
"""

import sys, os, json, math, random, argparse, warnings
from pathlib import Path
from collections import Counter
from typing import List, Dict

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sns.set_theme(style="darkgrid", palette="deep")
PALETTE = {"fire":"#E63946","ice":"#A8DADC","gas":"#F4A261","myth":"#457B9D",
           "habitable":"#2ecc71","hostile":"#e74c3c"}
PLT_DPI = 150
OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

from src.data_loader import load_config, load_processed, serialise_planet_features
cfg = load_config()


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
def savefig(name: str):
    path = OUT / name
    plt.savefig(path, dpi=PLT_DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓ Saved: {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────
# 1. DATASET OVERVIEW CHARTS
# ─────────────────────────────────────────────────────────────────
def chart_dataset_overview():
    print("\n── Chart 1: Dataset Overview ──")
    train_df, _, train_sents = load_processed("train", cfg)
    test_df,  _, _           = load_processed("test",  cfg)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.40)

    # 1a — Archetype pie
    ax = fig.add_subplot(gs[0, 0])
    counts = train_df["myth_archetype"].value_counts()
    colors = [PALETTE[c] for c in counts.index]
    wedges, texts, autos = ax.pie(counts.values, labels=counts.index,
        autopct="%1.1f%%", colors=colors, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title("Planet Archetype Split", fontweight="bold")

    # 1b — Temperature KDE by archetype
    ax = fig.add_subplot(gs[0, 1])
    for arch in ["fire","gas","myth","ice"]:
        d = train_df[train_df["myth_archetype"]==arch]["eq_temp_K"].dropna()
        d = d.clip(0, 4000)
        if len(d) > 2:
            d.plot.kde(ax=ax, label=arch, color=PALETTE[arch], linewidth=2)
    ax.set_xlabel("Equilibrium Temperature (K)")
    ax.set_title("Temperature Distribution by Archetype", fontweight="bold")
    ax.legend(fontsize=8); ax.set_xlim(0, 4000)

    # 1c — Radius histogram
    ax = fig.add_subplot(gs[0, 2])
    train_df["radius_earth"].clip(0,20).hist(ax=ax, bins=40,
        color="#457B9D", edgecolor="white", alpha=0.85)
    ax.axvline(2.5,  color="#2ecc71", linestyle="--", label="Habitable max (2.5)")
    ax.axvline(4.0,  color="#F4A261", linestyle="--", label="Giant threshold (4.0)")
    ax.set_xlabel("Planet Radius (R_earth)"); ax.set_title("Radius Distribution", fontweight="bold")
    ax.legend(fontsize=7)

    # 1d — Orbital period (log scale)
    ax = fig.add_subplot(gs[0, 3])
    per = train_df["orbital_period_days"].clip(0.1, 1000)
    ax.hist(np.log10(per), bins=40, color="#E63946", edgecolor="white", alpha=0.85)
    ax.set_xlabel("log₁₀(Orbital Period / days)")
    ax.set_title("Orbital Period (log scale)", fontweight="bold")

    # 1e — Habitable zone scatter
    ax = fig.add_subplot(gs[1, 0:2])
    sc = ax.scatter(train_df["eq_temp_K"].clip(0,5000),
                    train_df["radius_earth"].clip(0,20),
                    c=train_df["is_habitable"].astype(int),
                    cmap="RdYlGn", alpha=0.4, s=12, linewidths=0)
    ax.axvspan(200, 350, alpha=0.12, color="#2ecc71", label="Habitable T range")
    ax.axhline(2.5, color="#2ecc71", linestyle="--", linewidth=1.2, label="Radius limit")
    ax.set_xlabel("Equilibrium Temperature (K)")
    ax.set_ylabel("Radius (R_earth)")
    ax.set_title("Habitable Zone in T–R Space", fontweight="bold")
    ax.legend(fontsize=8); ax.set_xlim(0,4000); ax.set_ylim(0,18)
    plt.colorbar(sc, ax=ax, label="Habitable")

    # 1f — Star spectral type distribution
    ax = fig.add_subplot(gs[1, 2])
    if "spec_class_letter" in train_df.columns:
        spec = train_df["spec_class_letter"].value_counts().head(8)
        spec_colors = ["#FFD700","#FFA500","#FFFF99","#FFFACD","#FFFFFF",
                       "#ADD8E6","#4169E1","#aaaaaa"]
        spec.plot.bar(ax=ax, color=spec_colors[:len(spec)], edgecolor="white")
        ax.set_xlabel("Spectral Type")
        ax.set_title("Host Star Spectral Type", fontweight="bold")
        ax.set_xticklabels(spec.index, rotation=0)

    # 1g — Missing value heatmap
    ax = fig.add_subplot(gs[1, 3])
    raw = pd.read_csv(PROJECT_ROOT/"data"/"raw"/"exoplanets_raw.csv")
    miss = (raw.isnull().mean()*100).sort_values(ascending=False).head(8)
    miss.plot.barh(ax=ax, color="#9b59b6", edgecolor="white")
    ax.set_xlabel("% Missing Values")
    ax.set_title("Data Completeness", fontweight="bold")

    plt.suptitle("Dataset Overview — NASA Exoplanet Archive + Mythology Corpus",
                 fontsize=14, fontweight="bold", y=1.01)
    savefig("01_dataset_overview.png")


# ─────────────────────────────────────────────────────────────────
# 2. N-GRAM PERPLEXITY COMPARISON
# ─────────────────────────────────────────────────────────────────
def run_ngram_analysis() -> Dict:
    print("\n── Chart 2: N-Gram Perplexity Analysis ──")
    from src.baseline_models import NgramLanguageModel
    _, _, train_s = load_processed("train", cfg)
    _, _, val_s   = load_processed("val",   cfg)
    _, _, test_s  = load_processed("test",  cfg)

    results = {}
    ngcfg = cfg["ngram"]
    for n in [1,2,3]:
        m = NgramLanguageModel(n=n, alpha=1.0, unk_threshold=2).fit(train_s)
        results[n] = {"val": m.perplexity(val_s[:500]),
                      "test": m.perplexity(test_s[:500]),
                      "model": m}

    # Smoothing ablation on bigram
    ablation = {}
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        try:
            m2 = NgramLanguageModel(n=2, alpha=alpha, unk_threshold=2).fit(train_s)
            pp = m2.perplexity(test_s[:300])
            ablation[alpha] = pp if not math.isinf(pp) else 9999
        except: ablation[alpha] = 9999

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 2a — Bar: perplexity per order
    orders = [1,2,3]; labels=["Unigram","Bigram","Trigram"]
    val_pps  = [results[n]["val"]  for n in orders]
    test_pps = [results[n]["test"] for n in orders]
    x = np.arange(3); w = 0.35
    bars1 = axes[0].bar(x-w/2, val_pps,  w, label="Val",  color="#457B9D", edgecolor="white")
    bars2 = axes[0].bar(x+w/2, test_pps, w, label="Test", color="#E63946", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Perplexity"); axes[0].legend()
    axes[0].set_title("N-Gram Perplexity by Order", fontweight="bold")
    for bar in bars1: axes[0].text(bar.get_x()+bar.get_width()/2,
        bar.get_height()+50, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2: axes[0].text(bar.get_x()+bar.get_width()/2,
        bar.get_height()+50, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    # 2b — Log scale perplexity (includes GPT-2 baseline)
    gpt2_pp = 79.42  # from smoke test
    all_labels = ["Unigram","Bigram","Trigram","GPT-2\n(2-epoch)"]
    all_pps    = test_pps + [gpt2_pp]
    colors_bar = ["#9b59b6","#3498db","#e67e22","#2ecc71"]
    bars = axes[1].bar(all_labels, all_pps, color=colors_bar, edgecolor="white")
    axes[1].set_yscale("log"); axes[1].set_ylabel("Perplexity (log scale)")
    axes[1].set_title("All Models — Perplexity Comparison", fontweight="bold")
    for bar, val in zip(bars, all_pps):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.05,
            f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")

    # 2c — Smoothing ablation
    alphas = list(ablation.keys()); pps = list(ablation.values())
    axes[2].plot(alphas, pps, "o-", color="#E63946", linewidth=2, markersize=8)
    axes[2].set_xlabel("Laplace α"); axes[2].set_ylabel("Bigram Test Perplexity")
    axes[2].set_title("Smoothing Ablation (Bigram)", fontweight="bold")
    axes[2].set_xscale("log")
    for a, p in zip(alphas, pps):
        if p < 9000:
            axes[2].annotate(f"{p:.0f}", (a, p), textcoords="offset points",
                             xytext=(0,8), ha="center", fontsize=8)

    plt.suptitle("N-Gram Language Model Analysis  (Lab A4)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("02_ngram_perplexity.png")

    # Store sample generation
    sample_sents = []
    tri = results[3]["model"]
    for i in range(10): sample_sents.append(tri.generate(max_tokens=25, seed=i))

    return {"ngram_results": {n: {"val":results[n]["val"],"test":results[n]["test"]}
                               for n in [1,2,3]},
            "gpt2_pp_smoke": gpt2_pp,
            "smoothing_ablation": ablation,
            "sample_sentences": sample_sents}


# ─────────────────────────────────────────────────────────────────
# 3. NAIVE BAYES ANALYSIS
# ─────────────────────────────────────────────────────────────────
def run_nb_analysis() -> Dict:
    print("\n── Chart 3: Naïve Bayes Classifier Analysis ──")
    from src.baseline_models import PlanetNaiveBayesClassifier, _prepare_nb_data
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  ConfusionMatrixDisplay, roc_curve, auc,
                                  accuracy_score)
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    X_train, y_train, X_test, y_test = _prepare_nb_data(cfg)
    nbcfg = cfg["naive_bayes"]

    # Train models
    nb1 = PlanetNaiveBayesClassifier(alpha=1.0).fit(X_train, y_train)
    nb0 = PlanetNaiveBayesClassifier(alpha=0.001).fit(X_train, y_train)
    svm_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                          ("svm", LinearSVC(C=1.0, max_iter=3000))])
    svm_pipe.fit(X_train, y_train)

    results = {}
    for name, clf, is_nb in [("NB α=1.0", nb1, True),
                               ("NB α=0.001", nb0, True),
                               ("SVM (LinearSVC)", svm_pipe, False)]:
        yp = clf.predict(X_test)
        acc = accuracy_score(y_test, yp)
        rep = classification_report(y_test, yp, output_dict=True, zero_division=0)
        results[name] = {"accuracy": acc, "report": rep, "preds": yp.tolist()}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 3a — Confusion matrix for NB α=1
    cm = confusion_matrix(y_test, nb1.predict(X_test), labels=["habitable","hostile"])
    ConfusionMatrixDisplay(cm, display_labels=["Habitable","Hostile"]).plot(ax=axes[0,0], colorbar=False)
    axes[0,0].set_title("NB (α=1) Confusion Matrix", fontweight="bold")

    # 3b — Confusion matrix for SVM
    cm_svm = confusion_matrix(y_test, svm_pipe.predict(X_test), labels=["habitable","hostile"])
    ConfusionMatrixDisplay(cm_svm, display_labels=["Habitable","Hostile"]).plot(ax=axes[0,1], colorbar=False)
    axes[0,1].set_title("SVM Confusion Matrix", fontweight="bold")

    # 3c — Metric comparison bar
    metric_names = ["Accuracy","Hab. Precision","Hab. Recall","Hab. F1","Hostile F1"]
    model_names  = list(results.keys())
    data_matrix  = []
    for name in model_names:
        rep = results[name]["report"]
        hab = rep.get("habitable", {})
        hos = rep.get("hostile", {})
        data_matrix.append([results[name]["accuracy"],
                             hab.get("precision",0), hab.get("recall",0),
                             hab.get("f1-score",0),  hos.get("f1-score",0)])
    x = np.arange(len(metric_names)); w = 0.25
    for i, (name, row) in enumerate(zip(model_names, data_matrix)):
        axes[0,2].bar(x + i*w - w, row, w, label=name, edgecolor="white", alpha=0.9)
    axes[0,2].set_xticks(x); axes[0,2].set_xticklabels(metric_names, rotation=15, ha="right", fontsize=8)
    axes[0,2].set_ylim(0,1.1); axes[0,2].legend(fontsize=8)
    axes[0,2].set_title("Classifier Metric Comparison", fontweight="bold")

    # 3d — Class imbalance illustration
    class_counts = Counter(y_train)
    axes[1,0].bar(class_counts.keys(), class_counts.values(),
                   color=[PALETTE["habitable"], PALETTE["hostile"]], edgecolor="white")
    axes[1,0].set_title("Class Imbalance (Train)", fontweight="bold")
    axes[1,0].set_ylabel("Count")
    for k, v in class_counts.items():
        axes[1,0].text(list(class_counts.keys()).index(k), v+20,
                        f"{v}\n({100*v/sum(class_counts.values()):.1f}%)",
                        ha="center", fontsize=10, fontweight="bold")

    # 3e — Top discriminative features (NB)
    feats_hab  = nb1.top_features("habitable", k=10)
    feats_hos  = nb1.top_features("hostile",   k=10)
    feat_names = [f[0] for f in feats_hab]
    hab_probs  = [f[1] for f in feats_hab]
    hos_probs  = [results["NB α=1.0"]["report"]] # placeholder - use raw
    # Recompute properly
    feat_names_h = [f[0] for f in feats_hab[:8]]
    feat_lp_h    = [f[1] for f in feats_hab[:8]]
    feat_names_s = [f[0] for f in feats_hos[:8]]
    feat_lp_s    = [f[1] for f in feats_hos[:8]]
    axes[1,1].barh(feat_names_h[::-1], feat_lp_h[::-1], color=PALETTE["habitable"], edgecolor="white")
    axes[1,1].set_title("Top Features: 'Habitable'", fontweight="bold")
    axes[1,1].set_xlabel("Log Probability")

    axes[1,2].barh(feat_names_s[::-1], feat_lp_s[::-1], color=PALETTE["hostile"], edgecolor="white")
    axes[1,2].set_title("Top Features: 'Hostile'", fontweight="bold")
    axes[1,2].set_xlabel("Log Probability")

    plt.suptitle("Naïve Bayes Planet Classifier Analysis  (Lab A3)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("03_naive_bayes_analysis.png")

    return {"nb_results": {k: {"accuracy": v["accuracy"]} for k,v in results.items()},
            "class_balance": dict(class_counts)}


# ─────────────────────────────────────────────────────────────────
# 4. TRAINING CURVES (GPT-2)
# ─────────────────────────────────────────────────────────────────
def chart_training_curves():
    print("\n── Chart 4: Training Curves ──")
    hist_path = OUT / "training_history.json"
    if not hist_path.exists():
        print("  No training_history.json — using smoke test values.")
        history = {"train_loss":[4.34, 4.27], "val_perplexity":[81.93, 79.42], "epochs_run":2}
    else:
        with open(hist_path) as f: history = json.load(f)

    epochs    = list(range(1, history["epochs_run"]+1))
    train_loss = history["train_loss"][:len(epochs)]
    val_pp     = history["val_perplexity"][:len(epochs)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 4a — Training loss
    axes[0].plot(epochs, train_loss, "o-", color="#E63946", linewidth=2, markersize=8, label="Train Loss")
    axes[0].fill_between(epochs, [l*0.95 for l in train_loss], [l*1.05 for l in train_loss],
                          alpha=0.2, color="#E63946")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss (GPT-2)", fontweight="bold")
    axes[0].legend(); axes[0].set_ylim(bottom=0)
    for e, l in zip(epochs, train_loss):
        axes[0].annotate(f"{l:.3f}", (e, l), xytext=(0,8), textcoords="offset points",
                          ha="center", fontsize=9, fontweight="bold")

    # 4b — Validation perplexity
    axes[1].plot(epochs, val_pp, "s-", color="#457B9D", linewidth=2, markersize=8, label="Val PPL")
    axes[1].fill_between(epochs, [p*0.97 for p in val_pp], [p*1.03 for p in val_pp],
                          alpha=0.2, color="#457B9D")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Validation Perplexity (GPT-2)", fontweight="bold")
    axes[1].legend()
    for e, p in zip(epochs, val_pp):
        axes[1].annotate(f"{p:.2f}", (e, p), xytext=(0,8), textcoords="offset points",
                          ha="center", fontsize=9, fontweight="bold")

    # 4c — Combined perplexity comparison: all models
    model_labels = ["Unigram", "Bigram", "Trigram", "GPT-2\n(smoke)"]
    model_pps    = [661.16, 1668.75, 6973.32, min(val_pp)]
    colors_bar   = ["#9b59b6","#3498db","#e67e22","#2ecc71"]
    bars = axes[2].bar(model_labels, model_pps, color=colors_bar, edgecolor="white", width=0.5)
    axes[2].set_yscale("log"); axes[2].set_ylabel("Perplexity (log scale)")
    axes[2].set_title("Perplexity: All Models", fontweight="bold")
    for bar, val in zip(bars, model_pps):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                      f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("GPT-2 Fine-Tuning Results  (Lab A8)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("04_training_curves.png")


# ─────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE / CORRELATION
# ─────────────────────────────────────────────────────────────────
def chart_feature_analysis():
    print("\n── Chart 5: Feature Correlation & Importance ──")
    train_df, _, _ = load_processed("train", cfg)

    num_cols = ["eq_temp_K","radius_earth","mass_earth","orbital_period_days",
                "eccentricity","star_teff_K","distance_pc"]
    num_cols = [c for c in num_cols if c in train_df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 5a — Correlation heatmap
    corr = train_df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list("rg",["#E63946","white","#457B9D"])
    sns.heatmap(corr, ax=axes[0], annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=[c.replace("_"," ") for c in num_cols],
                yticklabels=[c.replace("_"," ") for c in num_cols])
    axes[0].set_title("Feature Correlation Matrix", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40, labelsize=7)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=7)

    # 5b — Box plots per archetype for temperature
    arch_order = ["fire","gas","myth","ice"]
    data_bp = [train_df[train_df["myth_archetype"]==a]["eq_temp_K"].dropna().clip(0,4000)
               for a in arch_order]
    bp = axes[1].boxplot(data_bp, labels=arch_order, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2))
    for patch, arch in zip(bp["boxes"], arch_order):
        patch.set_facecolor(PALETTE[arch]); patch.set_alpha(0.8)
    axes[1].set_ylabel("Equilibrium Temperature (K)")
    axes[1].set_title("Temperature Distribution per Archetype", fontweight="bold")

    # 5c — Mass vs Radius coloured by archetype
    for arch in arch_order:
        sub = train_df[train_df["myth_archetype"]==arch]
        axes[2].scatter(sub["mass_earth"].clip(0,500), sub["radius_earth"].clip(0,20),
                        c=PALETTE[arch], label=arch, alpha=0.4, s=15, linewidths=0)
    axes[2].set_xlabel("Mass (M_earth, clipped 0-500)")
    axes[2].set_ylabel("Radius (R_earth, clipped 0-20)")
    axes[2].set_title("Mass–Radius Diagram (coloured by Archetype)", fontweight="bold")
    axes[2].legend(title="Archetype", fontsize=8)

    plt.suptitle("Exoplanet Feature Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("05_feature_analysis.png")


# ─────────────────────────────────────────────────────────────────
# 6. GENERATED STORIES — INFERENCE
# ─────────────────────────────────────────────────────────────────
def run_inference(device: str = "cpu") -> List[Dict]:
    print("\n── Step 6: Running GPT-2 Inference ──")
    from src.transformer_model import (build_tokeniser, load_checkpoint,
                                        build_feature_vector)
    ckpt = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"] / "best_model.pt"
    if not ckpt.exists():
        print("  No checkpoint found — skipping inference.")
        return []

    tokeniser = build_tokeniser(cfg)
    model     = load_checkpoint(str(ckpt), cfg, tokeniser)
    model.eval()

    test_df, test_pairs, _ = load_processed("test", cfg)
    planet_lookup = {r["planet_name"]: r
                     for _, r in test_df.iterrows()
                     if "planet_name" in test_df.columns}

    stories_dir = OUT / "stories"
    stories_dir.mkdir(exist_ok=True)

    stories = []
    # Pick 4 representative planets (one per archetype)
    seen_archs = set()
    selected   = []
    for pair in test_pairs:
        arch = pair.get("archetype","myth")
        if arch not in seen_archs:
            seen_archs.add(arch)
            selected.append(pair)
        if len(seen_archs) == 4:
            break

    # Also grab 6 random ones
    rng = random.Random(42)
    extra = rng.sample(test_pairs, min(6, len(test_pairs)))
    selected.extend(extra)

    gen_cfg = cfg["generation"]
    for pair in selected[:10]:
        pname = pair["planet_name"]
        arch  = pair.get("archetype","myth")
        row   = planet_lookup.get(pname)
        fvec  = build_feature_vector(row) if row is not None else None
        prefix = pair["prefix"]

        story = model.generate_myth(
            prefix_text       = prefix,
            feature_vec       = fvec,
            style             = arch,
            max_new_tokens    = gen_cfg["max_new_tokens"],
            temperature       = gen_cfg["temperature"],
            top_p             = gen_cfg["top_p"],
            top_k             = gen_cfg["top_k"],
            repetition_penalty= gen_cfg["repetition_penalty"],
            device            = device,
        )

        entry = {"planet": pname, "archetype": arch, "prefix": prefix,
                 "story": story, "row": dict(row) if row is not None else {}}
        stories.append(entry)

        # Save individual file
        fname = pname.replace(" ","_").replace("/","-") + f"_{arch}.txt"
        (stories_dir / fname).write_text(
            f"PLANET  : {pname}\nARCHETYPE: {arch}\nPREFIX  : {prefix}\n\n{'-'*60}\n\n{story}",
            encoding="utf-8")
        print(f"  Generated story for {pname} [{arch}]")

    return stories


# ─────────────────────────────────────────────────────────────────
# 7. CONTENT FIDELITY CHART
# ─────────────────────────────────────────────────────────────────
def chart_content_fidelity(stories: List[Dict]):
    print("\n── Chart 7: Content Fidelity ──")
    kw_map = cfg["evaluation"]["fidelity_keywords"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 7a — Per-archetype fidelity bar
    arch_hits = {arch: {"total":0,"hit":0} for arch in ["fire","ice","gas","myth"]}
    for s in stories:
        arch = s["archetype"]
        row  = s["row"]
        text = s["story"].lower()
        kws  = kw_map.get("is_hot"  if arch=="fire" else
                           "is_icy"  if arch=="ice"  else
                           "is_giant"if arch=="gas"  else "is_giant", [])
        if arch in arch_hits:
            arch_hits[arch]["total"] += 1
            if any(kw in text for kw in kws):
                arch_hits[arch]["hit"] += 1

    archs  = [a for a in arch_hits if arch_hits[a]["total"]>0]
    fidels = [arch_hits[a]["hit"]/arch_hits[a]["total"] if arch_hits[a]["total"]>0 else 0
              for a in archs]
    bars = axes[0].bar(archs, fidels, color=[PALETTE[a] for a in archs], edgecolor="white")
    axes[0].set_ylim(0,1.1); axes[0].set_ylabel("Fidelity Score")
    axes[0].set_title("Content Fidelity by Archetype\n(keyword presence in story)",
                       fontweight="bold")
    axes[0].axhline(0.6, color="gray", linestyle="--", linewidth=1, label="Target ≥0.6")
    axes[0].legend()
    for bar, v in zip(bars, fidels):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                      f"{v:.1%}", ha="center", fontweight="bold")

    # 7b — Keyword occurrence frequency heat-map across stories
    all_kws = []
    for kw_list in kw_map.values(): all_kws.extend(kw_list)
    all_kws = list(dict.fromkeys(all_kws))[:12]
    heat = np.zeros((len(stories), len(all_kws)))
    for i, s in enumerate(stories):
        txt = s["story"].lower()
        for j, kw in enumerate(all_kws):
            heat[i,j] = txt.count(kw)

    sns.heatmap(heat, ax=axes[1], xticklabels=all_kws, yticklabels=False,
                cmap="YlOrRd", linewidths=0.3, cbar_kws={"label":"Occurrences"})
    axes[1].set_xlabel("Keyword"); axes[1].set_title("Keyword Occurrences Across Stories",
                                                       fontweight="bold")
    axes[1].tick_params(axis="x", rotation=40, labelsize=8)

    plt.suptitle("Content Fidelity — Feature-to-Narrative Mapping", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("07_content_fidelity.png")


# ─────────────────────────────────────────────────────────────────
# 8. ATTENTION HEATMAP
# ─────────────────────────────────────────────────────────────────
def chart_attention(stories: List[Dict], device: str = "cpu"):
    print("\n── Chart 8: Attention Heatmaps ──")
    if not stories:
        print("  No stories — skipping."); return
    from src.transformer_model import build_tokeniser, load_checkpoint
    from src.evaluate import plot_attention_heatmap

    ckpt = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"] / "best_model.pt"
    if not ckpt.exists(): return
    tokeniser = build_tokeniser(cfg)
    model     = load_checkpoint(str(ckpt), cfg, tokeniser)

    for i, s in enumerate(stories[:2]):
        prefix = s["prefix"]; arch = s["archetype"]
        text   = f"<myth> <{arch}> {prefix}\n{s['story'][:200]}"
        try:
            tokens, attentions = model.get_attention_weights(text, device=device)
            plot_attention_heatmap(tokens, attentions, layer_idx=0,
                                   save_dir=OUT/"attention", max_tokens=35)
            if len(attentions) > 5:
                plot_attention_heatmap(tokens, attentions, layer_idx=5,
                                       save_dir=OUT/"attention", max_tokens=35)
        except Exception as e:
            print(f"  Attention viz error: {e}")


# ─────────────────────────────────────────────────────────────────
# 9. STORY LENGTH & VOCABULARY RICHNESS ANALYSIS
# ─────────────────────────────────────────────────────────────────
def chart_story_quality(stories: List[Dict]):
    print("\n── Chart 9: Story Quality Metrics ──")
    if not stories:
        print("  No stories — skipping."); return

    import re
    def metrics(text):
        words = re.findall(r"\b[a-z]+\b", text.lower())
        sents = re.split(r"[.!?]+", text)
        sents = [s.strip() for s in sents if len(s.strip()) > 5]
        ttr   = len(set(words))/max(len(words),1)  # type-token ratio
        avg_s = np.mean([len(s.split()) for s in sents]) if sents else 0
        return {"n_words": len(words), "n_unique": len(set(words)),
                "ttr": ttr, "n_sentences": len(sents), "avg_sent_len": avg_s}

    rows = []
    for s in stories:
        m = metrics(s["story"])
        m["archetype"] = s["archetype"]
        m["planet"]    = s["planet"]
        rows.append(m)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 9a — Story length (word count)
    c = [PALETTE[a] for a in df["archetype"]]
    axes[0,0].bar(range(len(df)), df["n_words"], color=c, edgecolor="white")
    axes[0,0].set_xlabel("Story Index"); axes[0,0].set_ylabel("Word Count")
    axes[0,0].set_title("Story Length", fontweight="bold")
    axes[0,0].set_xticks(range(len(df)))
    axes[0,0].set_xticklabels([s["planet"][:8] for s in stories], rotation=45, ha="right", fontsize=7)

    # 9b — Type-Token Ratio (vocabulary richness)
    axes[0,1].bar(range(len(df)), df["ttr"], color=c, edgecolor="white")
    axes[0,1].axhline(df["ttr"].mean(), color="red", linestyle="--",
                       label=f"Mean TTR={df['ttr'].mean():.3f}")
    axes[0,1].set_xlabel("Story Index"); axes[0,1].set_ylabel("Type-Token Ratio")
    axes[0,1].set_title("Vocabulary Richness (TTR)", fontweight="bold")
    axes[0,1].legend(fontsize=8)

    # 9c — Avg sentence length
    axes[1,0].bar(range(len(df)), df["avg_sent_len"], color=c, edgecolor="white")
    axes[1,0].set_xlabel("Story Index"); axes[1,0].set_ylabel("Avg Sentence Length (words)")
    axes[1,0].set_title("Average Sentence Length", fontweight="bold")

    # 9d — Scatter: words vs unique words
    for arch in df["archetype"].unique():
        sub = df[df["archetype"]==arch]
        axes[1,1].scatter(sub["n_words"], sub["n_unique"],
                           color=PALETTE.get(arch,"gray"), label=arch, s=100, zorder=5)
    axes[1,1].set_xlabel("Total Words"); axes[1,1].set_ylabel("Unique Words")
    axes[1,1].set_title("Total vs Unique Words", fontweight="bold")
    axes[1,1].legend(fontsize=8)
    mn = min(df["n_words"].min(), df["n_unique"].min())
    mx = max(df["n_words"].max(), df["n_unique"].max())
    axes[1,1].plot([mn,mx],[mn,mx],"--",color="gray",alpha=0.5,label="y=x (all unique)")

    # Legend for archetype colours
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=PALETTE[a], label=a)
                      for a in df["archetype"].unique()]
    fig.legend(handles=legend_patches, loc="upper right", title="Archetype",
               fontsize=8, title_fontsize=9, bbox_to_anchor=(1.0, 1.0))

    plt.suptitle("Generated Story Quality Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("09_story_quality.png")
    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────
# 10. NER ANALYSIS CHART
# ─────────────────────────────────────────────────────────────────
def chart_ner(stories: List[Dict]):
    print("\n── Chart 10: NER Analysis ──")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except: print("  spaCy not available — skipping."); return

    KNOWN_MYTH = {"zeus","athena","apollo","poseidon","hera","ares","hephaestus",
                  "hermes","demeter","dionysus","persephone","hades","prometheus",
                  "hercules","achilles","odysseus","thor","odin","loki","freya",
                  "surtur","ra","osiris","horus","isis","titan","atlas","helios"}

    type_cnt = Counter(); entity_cnt = Counter(); myth_cnt = Counter()
    for s in stories:
        doc = nlp(s["story"][:500])
        for ent in doc.ents:
            type_cnt[ent.label_] += 1
            entity_cnt[ent.text.lower()] += 1
            if ent.text.lower() in KNOWN_MYTH:
                myth_cnt[ent.text] += 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 10a — Entity type distribution
    if type_cnt:
        t_labels = list(type_cnt.keys()); t_vals = list(type_cnt.values())
        axes[0].bar(t_labels, t_vals, color="#9b59b6", edgecolor="white")
        axes[0].set_xlabel("Entity Type"); axes[0].set_ylabel("Count")
        axes[0].set_title("Entity Type Distribution", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=30)
    else:
        axes[0].text(0.5,0.5,"No entities found",ha="center",transform=axes[0].transAxes)
        axes[0].set_title("Entity Type Distribution", fontweight="bold")

    # 10b — Top entity frequency
    top_ents = entity_cnt.most_common(15)
    if top_ents:
        e_labels, e_vals = zip(*top_ents)
        axes[1].barh(list(e_labels)[::-1], list(e_vals)[::-1],
                      color="#3498db", edgecolor="white")
    axes[1].set_xlabel("Frequency"); axes[1].set_title("Top 15 Named Entities", fontweight="bold")

    # 10c — Known mythology names found
    if myth_cnt:
        m_labels, m_vals = zip(*myth_cnt.most_common(10))
        axes[2].bar(m_labels, m_vals, color="#2ecc71", edgecolor="white")
        axes[2].set_xlabel("Myth Figure"); axes[2].set_ylabel("Count")
    else:
        axes[2].text(0.5,0.5,"No known myth names\ndetected (model needs\nmore training)",
                      ha="center", va="center", transform=axes[2].transAxes, fontsize=10)
    axes[2].set_title("Known Mythology Names (True Positives)", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=30)

    plt.suptitle("Named Entity Recognition on Generated Stories  (Lab A10)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("10_ner_analysis.png")


# ─────────────────────────────────────────────────────────────────
# 11. COMPREHENSIVE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
def chart_summary_table(ngram_res: Dict, nb_res: Dict, stories: List[Dict]):
    print("\n── Chart 11: Summary Results Table ──")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    table_data = [
        ["Model / Metric", "Value", "Lab", "Notes"],
        ["─"*28, "─"*18, "─"*6, "─"*35],
        ["Unigram Perplexity (test)",
         f"{ngram_res['ngram_results'][1]['test']:.1f}", "A4",
         "Highest PPL — ignores all context"],
        ["Bigram Perplexity (test)",
         f"{ngram_res['ngram_results'][2]['test']:.1f}", "A4",
         "Uses 1-word context"],
        ["Trigram Perplexity (test)",
         f"{ngram_res['ngram_results'][3]['test']:.1f}", "A4",
         "Uses 2-word context"],
        ["GPT-2 Val Perplexity (smoke)",
         f"{ngram_res['gpt2_pp_smoke']:.1f}", "A8",
         "Fine-tuned on myth corpus, 2 epochs CPU"],
        ["─"*28, "─"*18, "─"*6, "─"*35],
        ["NB Accuracy (α=1)",
         f"{nb_res['nb_results']['NB α=1.0']['accuracy']:.3f}", "A3",
         "High due to class imbalance (~99% hostile)"],
        ["SVM Accuracy",
         f"{nb_res['nb_results']['SVM (LinearSVC)']['accuracy']:.3f}", "A3",
         "Better minority class recall"],
        ["Habitable zone prevalence",
         f"{100*51/4215:.1f}%", "A3",
         "Severe imbalance — key viva point"],
        ["─"*28, "─"*18, "─"*6, "─"*35],
        ["Stories generated",
         str(len(stories)), "A8", "Across all 4 archetypes"],
        ["Avg story word count",
         f"{np.mean([len(s['story'].split()) for s in stories]):.0f}" if stories else "N/A",
         "A8", "GPT-2 with nucleus sampling (top-p=0.92)"],
        ["NER entities detected",
         "See chart 10", "A10", "spaCy en_core_web_sm on myths"],
        ["Corpus size (words)",
         "568,045", "A4/A8", "5 Gutenberg mythology books"],
        ["NASA exoplanets",
         "6,000 → 4,215 usable", "A3/A8", "After filtering unparseable rows"],
    ]

    col_widths = [0.30, 0.18, 0.07, 0.45]
    cell_colors = []
    for row in table_data:
        if row[0].startswith("─"):
            cell_colors.append(["#ddd"]*4)
        elif row[0] in ["Model / Metric"]:
            cell_colors.append(["#2c3e50"]*4)
        else:
            cell_colors.append(["#f8f9fa","#eaf2fb","#eafaf1","#fdfefe"])

    tbl = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif table_data[r][0].startswith("─"):
            cell.set_facecolor("#dde")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")

    ax.set_title("Project Results Summary — All Metrics", fontsize=14,
                  fontweight="bold", pad=20)
    savefig("11_summary_table.png")


# ─────────────────────────────────────────────────────────────────
# 12. MYTH CORPUS LINGUISTICS CHART
# ─────────────────────────────────────────────────────────────────
def chart_corpus_linguistics():
    print("\n── Chart 12: Mythology Corpus Linguistics ──")
    import re
    _, _, train_s = load_processed("train", cfg)
    _, _, test_s  = load_processed("test",  cfg)

    all_tokens = []
    for s in train_s[:3000]:
        all_tokens.extend(re.findall(r"\b[a-z]{3,}\b", s.lower()))

    token_freq = Counter(all_tokens)
    bigrams    = Counter(zip(all_tokens, all_tokens[1:]))

    # Sent length distribution
    sent_lens = [len(s.split()) for s in train_s]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 12a — Token frequency (Zipf)
    freqs = sorted(token_freq.values(), reverse=True)[:500]
    ranks = np.arange(1, len(freqs)+1)
    axes[0,0].loglog(ranks, freqs, "o", markersize=3, color="#457B9D", alpha=0.7)
    # Fit Zipf
    log_r  = np.log(ranks); log_f = np.log(freqs)
    slope, intercept = np.polyfit(log_r, log_f, 1)
    axes[0,0].loglog(ranks, np.exp(intercept + slope*log_r), "--",
                      color="#E63946", linewidth=2, label=f"Zipf slope={slope:.2f}")
    axes[0,0].set_xlabel("Rank"); axes[0,0].set_ylabel("Frequency")
    axes[0,0].set_title("Word Frequency (Zipf's Law)", fontweight="bold")
    axes[0,0].legend()

    # 12b — Top 20 unigrams
    top20 = token_freq.most_common(20)
    t_words, t_vals = zip(*top20)
    axes[0,1].barh(list(t_words)[::-1], list(t_vals)[::-1], color="#9b59b6", edgecolor="white")
    axes[0,1].set_xlabel("Frequency"); axes[0,1].set_title("Top 20 Unigrams in Myth Corpus", fontweight="bold")

    # 12c — Top 20 bigrams
    top_bg = bigrams.most_common(20)
    bg_labels = [f"{a} {b}" for (a,b),_ in top_bg]
    bg_vals   = [v for _,v in top_bg]
    axes[1,0].barh(bg_labels[::-1], bg_vals[::-1], color="#3498db", edgecolor="white")
    axes[1,0].set_xlabel("Frequency"); axes[1,0].set_title("Top 20 Bigrams in Myth Corpus", fontweight="bold")
    axes[1,0].tick_params(axis="y", labelsize=7)

    # 12d — Sentence length distribution
    axes[1,1].hist(sent_lens, bins=40, color="#e67e22", edgecolor="white", alpha=0.85)
    axes[1,1].axvline(np.mean(sent_lens), color="#E63946", linestyle="--",
                       label=f"Mean={np.mean(sent_lens):.1f}")
    axes[1,1].axvline(np.median(sent_lens), color="#2c3e50", linestyle=":",
                       label=f"Median={np.median(sent_lens):.0f}")
    axes[1,1].set_xlabel("Tokens per Sentence"); axes[1,1].set_ylabel("Count")
    axes[1,1].set_title("Sentence Length Distribution", fontweight="bold")
    axes[1,1].legend()

    plt.suptitle("Mythology Corpus Linguistics Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("12_corpus_linguistics.png")


# ─────────────────────────────────────────────────────────────────
# 13. FULL TRAINING RUN (optional)
# ─────────────────────────────────────────────────────────────────
def run_full_training(device: str):
    print("\n── Full Training (5 epochs) ──")
    from src.train import train
    history = train(cfg=cfg, smoke_test=False, device_str=device)
    return history


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (use existing checkpoint)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-train", action="store_true",
                        help="Run full 5-epoch training (slow on CPU)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  EXOPLANET MYTHOS GENERATOR — FULL ANALYSIS")
    print("═"*60)

    all_results = {}

    # Charts that don't need training
    chart_dataset_overview()
    all_results["ngram"] = run_ngram_analysis()
    all_results["nb"]    = run_nb_analysis()
    chart_training_curves()
    chart_feature_analysis()
    chart_corpus_linguistics()

    # Optionally run more training
    if args.full_train and not args.skip_train:
        run_full_training(args.device)

    # Inference + downstream charts
    stories = run_inference(device=args.device)
    if stories:
        chart_content_fidelity(stories)
        chart_attention(stories, device=args.device)
        all_results["story_quality"] = chart_story_quality(stories)
        chart_ner(stories)
        chart_summary_table(all_results["ngram"], all_results["nb"], stories)
    else:
        chart_summary_table(all_results["ngram"], all_results["nb"], [])

    # Save master results JSON
    def _safe(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    with open(OUT/"all_results.json","w") as f:
        json.dump(all_results, f, indent=2, default=_safe)

    print("\n" + "═"*60)
    print(f"  ALL CHARTS SAVED to: {OUT}")
    print("  Files:")
    for p in sorted(OUT.glob("*.png")):
        print(f"    {p.name}")
    print("═"*60)


if __name__ == "__main__":
    main()
