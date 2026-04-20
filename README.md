# Exoplanet Mythos Generator

> **CST-106 Applications of NLP** · Project `whispering-h2`  
> Translating NASA exoplanet data into mythological narratives using classical and neural NLP.

---

## Overview

The **Exoplanet Mythos Generator** takes physical parameters of confirmed exoplanets from the NASA Exoplanet Archive and generates rich, myth-style narrative stories conditioned on each planet's astrophysical fingerprint.

A planet with an equilibrium temperature of **1 740 K** and a radius of **11 Re** is automatically classified as a *fire archetype* and the fine-tuned GPT-2 model generates a story involving forge gods, molten seas, and eternal flame — grounded in the planet's actual science.

### Lab Topic Coverage

| Lab | Topic | Module | Implementation |
|-----|-------|--------|----------------|
| A3 | Bag-of-Words + Multinomial Naïve Bayes | `baseline_models.py` | Habitable/hostile planet classifier via `CountVectorizer` + `MultinomialNB` |
| A4 | N-gram Language Models | `baseline_models.py` | Unigram/bigram/trigram with Laplace smoothing; perplexity evaluation |
| A7 | Transformer Self-Attention | `transformer_model.py` + `evaluate.py` | GPT-2 attention weight extraction + per-head heatmap visualisation |
| A8 | Fine-tuning Pretrained Transformer | `transformer_model.py` + `train.py` | GPT-2 fine-tuned on myth corpus conditioned on planet features |
| A10 | Named Entity Recognition | `ner_analysis.py` | spaCy NER on generated stories; entity frequency + TP/FP analysis |

---

## Project Structure

```
whispering-h2/
├── README.md                   ← This file
├── requirements.txt            ← All dependencies (pinned)
├── config.yaml                 ← Master hyperparameter configuration
├── data/
│   ├── raw/                    ← Downloaded NASA + Gutenberg files (git-ignored)
│   └── processed/              ← Cleaned CSVs, pairs, sentence splits
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← NASA TAP fetch, myth corpus, preprocessing, splits
│   ├── baseline_models.py      ← N-gram LM (A4) + BoW/NB classifier (A3)
│   ├── transformer_model.py    ← GPT-2 wrapper + MLP latent encoder (A7/A8)
│   ├── train.py                ← Fine-tuning loop (AdamW, early stopping, FP16)
│   ├── evaluate.py             ← Full evaluation suite (PPL, BLEU, ROUGE, fidelity)
│   ├── generate.py             ← Myth generation CLI
│   └── ner_analysis.py         ← spaCy NER postprocessing (A10)
├── notebooks/
│   ├── exploration.py          ← Data visualisation and corpus statistics
│   └── attention_viz.py        ← Attention heatmap analysis (Lab A7)
├── outputs/
│   ├── stories/                ← Generated myth stories (.txt)
│   ├── attention/              ← Attention heatmap PNGs
│   ├── training_curve.png      ← Loss + perplexity curves
│   ├── nb_confusion_matrix.png ← NB classifier confusion matrix
│   ├── baseline_results.json   ← N-gram + NB numeric results
│   └── evaluation_results.json ← Full evaluation metrics
└── checkpoints/
    ├── best_model.pt           ← Best fine-tuned checkpoint
    └── epoch_XX.pt             ← Per-epoch checkpoints
```

---

## Quick Start

### 1. Install dependencies

```bash
cd /home/pradyuman/whispering-h2
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download data and preprocess

```bash
python src/data_loader.py
```

This fetches ~6 000 exoplanets from the NASA Exoplanet Archive via ADQL/TAP, downloads mythology texts from Project Gutenberg (Bulfinch, Homer, Norse legends), engineers features, builds planet→story pairs, and saves everything to `data/processed/`.

Expected output:
```
Planet splits — Train: 4384  Val: 548  Test: 548
Sentence splits — Train: 45000  Val: 2500  Test: 2500
Built 10000 planet→story pairs.
```

### 3. Run baseline models (Labs A3 + A4)

```bash
python src/baseline_models.py --test
```

Running this prints:
- N-gram perplexity for unigram / bigram / trigram on the mythology test set
- Smoothing ablation (α=0 vs α=1) demonstrating why Laplace smoothing is necessary
- NB classifier accuracy, F1, confusion matrix
- Top-10 discriminative features per class
- SVM baseline comparison

### 4. Fine-tune the Transformer (Labs A7 + A8)

```bash
# Fast smoke test (2 epochs, 50 pairs, CPU — for verification)
python src/train.py --smoke_test

# Full training (GPU recommended)
python src/train.py --epochs 5 --batch_size 16
```

Checkpoints are saved to `checkpoints/` after each epoch. Best model is saved as `checkpoints/best_model.pt`.

### 5. Generate myths

```bash
# By planet name
python src/generate.py --planet "55 Cnc e" --style fire

# By planet name with default style (auto-detected from archetype)
python src/generate.py --planet "Kepler-442b"

# Manual features (hypothetical planet)
python src/generate.py --temp 250 --radius 1.8 --mass 5 --style myth

# Batch: generate stories for 10 test planets
python src/generate.py --batch 10

# List available style tokens
python src/generate.py --list-styles
```

### 6. Full evaluation suite

```bash
python src/evaluate.py
```

Outputs `outputs/evaluation_results.json` with all metrics plus attention heatmaps in `outputs/attention/`.

### 7. NER analysis

```bash
python src/ner_analysis.py --n_samples 20
```

---

## Model Architecture

```
[Planet Features]  →  [2-layer MLP Encoder]  →  [Latent ∈ R^128]
                                                          │ (additive injection)
[Feature Prefix Text]  →  [GPT-2 Tokeniser]  →  [Token Embeddings ∈ R^768]
                                                          │
                               ┌──────────────────────────┘
                               ↓
                    ┌─────────────────────────────────────────┐
                    │      GPT-2 Transformer  (12 layers)      │
                    │  Each layer:                              │
                    │    LayerNorm → MHA (12 heads) → residual  │
                    │    LayerNorm → FFN (d=3072)  → residual   │
                    └─────────────────────────────────────────┘
                               │
                    [LM Head — Linear(768 → |V|)]
                               │
                    [Softmax → next-token distribution]
                               │
                    [Sampled myth story tokens]
```

**Feature Prefix Example** (input to GPT-2):
```
<myth> <fire> <FEAT> Temp=1740K | Radius=11.2Re | Mass=320Me | Period=3.1d |
Ecc=0.02 | Star=G | Dist=420pc | Hot=yes | Giant=yes | Habitable=no <PLANET>
```
The model generates everything after `<PLANET>`.

---

## Configuration

All hyperparameters live in `config.yaml`.  Key values:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model_name` | `gpt2` | Pretrained GPT-2 (117M) |
| `learning_rate` | `1e-5` | Fine-tuning range; prevents catastrophic forgetting |
| `batch_size` | `16` | GPU memory / gradient quality tradeoff |
| `max_length` | `512` | GPT-2 context window |
| `latent_dim` | `128` | MLP encoder output; injected additively into embeddings |
| `smoothing_alpha` | `1.0` | Laplace add-one smoothing for n-gram model |
| `warmup_steps` | `100` | LR ramped from 0 to target over 100 steps |
| `early_stopping_patience` | `2` | Stop after 2 epochs without val PPL improvement |
| `temperature` | `0.85` | Sampling temperature for generation |
| `top_p` | `0.92` | Nucleus sampling cutoff |

---

## Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| NASA Exoplanet Archive | TAP endpoint (ADQL, `ps` table) | ~5 500 planets | Feature extraction, NB classification |
| Bulfinch's Age of Fable | Project Gutenberg #22381 | ~300k words | Myth corpus |
| Myths of Ancient Greece and Rome | Project Gutenberg #14915 | ~250k words | Myth corpus |
| The Odyssey (Homer) | Project Gutenberg #1727 | ~200k words | Myth corpus |
| Iliad (Homer) | Project Gutenberg #1728 | ~200k words | Myth corpus |
| Norse Mythology | Project Gutenberg #348 | ~150k words | Myth corpus |

**Splits**: Planets 80/10/10 (stratified by archetype). Myth sentences 90/5/5.

---

## Evaluation Metrics

| Metric | Model | Expected Range | Lab |
|--------|-------|---------------|-----|
| Perplexity | Unigram | 800–1200 | A4 |
| Perplexity | Bigram | 150–300 | A4 |
| Perplexity | Trigram | 80–180 | A4 |
| Perplexity | GPT-2 fine-tuned | 40–60 | A8 |
| BLEU-4 | GPT-2 vs references | 5–20 | A8 |
| ROUGE-L | GPT-2 vs references | 0.15–0.40 | A8 |
| NB Accuracy | Habitable/hostile | 88–94% | A3 |
| Habitable F1 | NB (α=1) | 0.40–0.60 | A3 |
| Content Fidelity | GPT-2 | 0.60–0.90 | Custom |
| Myth Style Fraction | SVM | >0.70 | Custom |
| NER F1 | spaCy on stories | 0.50–0.75 | A10 |

---

## Key Design Decisions (Viva Notes)

### Why GPT-2 and not GPT-3/4?
GPT-2 is fully open-source, runs locally on a single GPU (or even CPU for smoke tests), and has publicly available weights.  GPT-3/4 require API access (paid), making reproducible research difficult.  GPT-2 Medium (345M) is large enough to produce coherent mythological prose after fine-tuning.

### Why text prefix conditioning and not cross-attention?
Modifying GPT-2's architecture to add cross-attention layers would require retraining from scratch and is complex to implement correctly.  Prepending features as a text prefix uses GPT-2 in its original form — the model attends to the prefix naturally via its causal self-attention.  The additive MLP latent injection provides a complementary continuous signal at the first layer.

### Why Laplace smoothing for n-grams?
Without smoothing, any n-gram in the test set that was never seen in training receives P=0, making the entire corpus perplexity infinite.  Laplace smoothing (α=1) assigns a small non-zero probability to all vocabulary entries, enabling meaningful perplexity computation on held-out text.

### Why stratified splitting for planets?
The archetype distribution is unbalanced (~40% fire, ~25% gas, ~25% myth, ~10% ice).  Without stratification, the test set might contain no ice-archetype planets at all, making it impossible to evaluate performance on that class.  Stratified sampling preserves the distribution in all splits.

### Why nucleus (top-p) sampling?
Greedy decoding collapses to repetitive, safe text.  Beam search optimises joint probability and is unsuitable for creative generation.  Nucleus sampling (Holtzman et al. 2020) restricts sampling to the dynamic top-p probability mass, proven to produce the most creative and non-repetitive text for language generation.

---

## Dependencies

```
torch>=2.1.0              # Core DL framework
transformers>=4.38.0      # GPT-2 model + tokeniser
datasets>=2.18.0          # HuggingFace datasets
scikit-learn>=1.4.0       # CountVectorizer, MultinomialNB, SVM
nltk>=3.8.1               # Tokenisation utilities
spacy>=3.7.0              # NER model (en_core_web_sm)
pandas>=2.2.0             # Exoplanet data handling
sacrebleu>=2.4.0          # BLEU evaluation
rouge-score>=0.1.2        # ROUGE evaluation
matplotlib>=3.8.0         # Visualisations
seaborn>=0.13.0           # Heatmaps (attention viz)
requests>=2.31.0          # NASA TAP API calls
PyYAML>=6.0.1             # Config loading
```

---

## References

1. NASA Exoplanet Archive TAP endpoint: `https://exoplanetarchive.ipac.caltech.edu/TAP`
2. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
3. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2).* OpenAI.
4. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularisation (AdamW).* ICLR.
5. Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration (nucleus sampling).* ICLR.
6. Wiseman et al. (2017). *Challenges in Data-to-Document Generation.* EMNLP.
7. Vig & Belinkov (2019). *Analyzing the Structure of Attention in a Transformer Language Model.* ACL.
8. Clark et al. (2019). *What Does BERT Look at? An Analysis of BERT's Attention.* ACL.
9. Bulfinch, T. (1855). *The Age of Fable.* Project Gutenberg #22381.
10. Homer (trans. Butler, 1900). *The Odyssey.* Project Gutenberg #1727.

---

*CST-106 Applications of NLP · whispering-h2 · April 2026*
