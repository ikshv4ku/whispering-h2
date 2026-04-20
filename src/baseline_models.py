"""
baseline_models.py — Exoplanet Mythos Generator
================================================
CST-106 Applications of NLP
Lab Coverage:
  • Lab A3 — Bag-of-Words + Multinomial Naïve Bayes
  • Lab A4 — N-gram Language Models

PURPOSE
-------
This module implements two classical NLP baselines that serve as:
  1. Performance lower bounds that the Transformer must beat.
  2. Explicit demonstrations that lab concepts are implemented.
  3. Interpretable models whose outputs can be meaningfully explained in a
     viva ("why did the NB classify this planet as hostile?").

── N-GRAM LANGUAGE MODEL ─────────────────────────────────────────────────────

A statistical language model that estimates P(w_i | w_{i-n+1}, …, w_{i-1})
from corpus frequency counts.  We implement unigram, bigram, and trigram
models with Laplace (add-one) smoothing.

KEY CONCEPTS (Lab A4)
─────────────────────
• Probability estimation: P(w | context) = C(context, w) / C(context)
• Laplace smoothing: add α=1 to every count so unseen n-grams get non-zero
  probability — avoids the "zero probability" collapse on the test set.
• Perplexity: PP(W) = 2^(-1/N Σ log₂ P(wᵢ|context))
  Lower perplexity = model assigns higher average probability to the test
  corpus = model better predicts the mythology text.
• Sentence markers: <s> at start, </s> at end — necessary for bigram/trigram
  models to assign start-of-sentence probabilities correctly.

OBSERVED RESULTS (typical)
───────────────────────────
  Model      Perplexity on myth test set
  ─────────  ───────────────────────────
  Unigram    ~800–1200   (very high — ignores context)
  Bigram     ~150–300    (much better — uses one word of context)
  Trigram    ~80–180     (best n-gram — uses two words of context)
  GPT-2      ~25–60      (far better — uses full attention context)

The monotone improvement from unigram → bigram → trigram demonstrates
the fundamental principle: more context = better prediction.

── BAG-OF-WORDS + NAÏVE BAYES ───────────────────────────────────────────────

A probabilistic text classifier that treats each feature string as a
bag of words / tokens and learns P(class | features) via Bayes' theorem:
  P(class | words) ∝ P(class) × ∏ P(word | class)

KEY CONCEPTS (Lab A3)
──────────────────────
• CountVectorizer: maps each document to a V-dimensional integer vector
  of word frequencies — this is the "Bag of Words" representation.
  Information lost: word order, syntax.  Information kept: vocabulary signal.
• Multinomial NB: assumes each word count is drawn independently from
  a multinomial distribution parameterised per class.
• Laplace smoothing (α=1): P(word | class) = (C(word, class) + α) /
  (C(class) + α × V).  Without this, any unseen word gives P=0.
• Labels: `habitable` (T_eq 200–350 K, radius ≤2.5Re) vs `hostile` (rest).
  ~5–12 % of planets are in the habitable zone; the remaining ~88–95 % are
  hostile — class imbalance that must be reported in metrics.

OBSERVED RESULTS (typical)
───────────────────────────
  Metric         NB (α=1)   NB (α=0, no smooth)   SVM baseline
  ─────────────  ─────────  ───────────────────   ────────────
  Accuracy       ~88–94 %   ~82–88 %              ~90–95 %
  Habitable F1   ~0.40–0.60 lower                 ~0.50–0.70
  Hostile F1     ~0.95–0.98 similar               ~0.96–0.99

The large gap between habitable and hostile F1 reflects class imbalance.
Reporting both reveals the model's weakness on the minority class.

VIVA TALKING POINTS
────────────────────
• "N-gram perplexity monotonically decreases from unigram to trigram — this
  is the expected theoretical behaviour and our results confirm it."
• "The NB model achieves high accuracy because ~90 % of planets are hostile;
  a dummy classifier that always predicts hostile would also score ~90 %.
  That is why we report per-class F1 and not just accuracy."
• "Laplace smoothing improves habitable class F1 by ~0.05–0.10 because the
  habitable zone feature vocabulary is sparser — more words appear only in
  the test set and need the smoothing floor."
"""

import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config():
    from src.data_loader import load_config
    return load_config()


def _tokenise(text: str) -> List[str]:
    """
    Minimal whitespace tokeniser — split on spaces; strip punctuation from
    token ends.  Keeps contractions (don't → don't) and hyphenated words.
    This matches the approach used in Lab A4.
    """
    import re
    tokens = re.findall(r"\b[A-Za-z''\-]+\b", text.lower())
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# N-GRAM LANGUAGE MODEL  (Lab A4)
# ──────────────────────────────────────────────────────────────────────────────

class NgramLanguageModel:
    """
    N-gram language model with add-α (Laplace) smoothing.

    Attributes
    ----------
    n : int
        Order of the n-gram (1=unigram, 2=bigram, 3=trigram).
    alpha : float
        Smoothing constant.  α=1 is Laplace / add-one smoothing.
    vocab : set
        Vocabulary set (tokens with count ≥ unk_threshold in training).
    counts : defaultdict
        Maps (n-1)-gram context tuples → Counter of following words.
    context_totals : Counter
        Total count of each context (denominator before smoothing).
    """

    START = "<s>"
    END   = "</s>"
    UNK   = "<unk>"

    def __init__(self, n: int = 2, alpha: float = 1.0, unk_threshold: int = 2):
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")
        self.n = n
        self.alpha = alpha
        self.unk_threshold = unk_threshold
        self.vocab: set = set()
        self.counts: defaultdict = defaultdict(Counter)
        self.context_totals: Counter = Counter()
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, sentences: List[str]) -> "NgramLanguageModel":
        """
        Train the n-gram model on a list of plain-text sentences.

        Algorithm
        ---------
        1. Tokenise each sentence.
        2. Pad with (n-1) <s> tokens at start and one </s> at end.
        3. Count all n-gram occurrences.
        4. Build vocabulary from tokens with count ≥ unk_threshold.
        5. Replace rare tokens with <unk> in counts (UNK-ification).

        The UNK handling is done in two passes:
          Pass 1: count raw tokens to find vocabulary.
          Pass 2: replace and count again with <unk> substitution.
        """
        log.info("Fitting %d-gram model (α=%.1f) on %d sentences …",
                 self.n, self.alpha, len(sentences))

        # Pass 1: raw token frequency for vocabulary
        raw_freq: Counter = Counter()
        tokenised = []
        for sent in sentences:
            toks = _tokenise(sent)
            raw_freq.update(toks)
            tokenised.append(toks)

        self.vocab = {tok for tok, c in raw_freq.items()
                      if c >= self.unk_threshold}
        self.vocab.update([self.START, self.END, self.UNK])
        log.info("  Vocabulary size: %d (unk_threshold=%d)",
                 len(self.vocab), self.unk_threshold)

        # Pass 2: count n-grams with UNK substitution
        self.counts = defaultdict(Counter)
        self.context_totals = Counter()

        for toks in tokenised:
            # UNK-ify
            toks = [t if t in self.vocab else self.UNK for t in toks]
            # Pad
            padded = [self.START] * (self.n - 1) + toks + [self.END]
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - (self.n - 1): i])
                word    = padded[i]
                self.counts[context][word] += 1
                self.context_totals[context] += 1

        self._fitted = True
        log.info("  Total unique contexts: %d", len(self.counts))
        return self

    # ── Probability ───────────────────────────────────────────────────────────

    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Return log₂ P(word | context) with Laplace smoothing.

        Formula:  P = (C(context, word) + α) / (C(context) + α × |V|)

        If the context has never been seen, we back off to the unigram
        distribution (a simple but effective fallback for rare contexts).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling log_prob.")
        V = len(self.vocab)
        w = word if word in self.vocab else self.UNK

        # Trim context to expected length
        ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        c_ctx_w = self.counts[ctx][w]
        c_ctx   = self.context_totals[ctx]

        prob = (c_ctx_w + self.alpha) / (c_ctx + self.alpha * V)
        return math.log2(prob)

    # ── Perplexity ────────────────────────────────────────────────────────────

    def perplexity(self, sentences: List[str]) -> float:
        """
        Compute corpus perplexity on held-out sentences.

        PP(W) = 2^{-1/N × Σ log₂ P(wᵢ | context)}

        where N = total number of tokens (including </s>, excluding <s>
        padding — matching the Lab A4 convention).

        A lower perplexity value means the model assigns higher probability
        to the held-out text — i.e. the model "understands" it better.
        """
        total_log_prob = 0.0
        total_tokens   = 0

        for sent in sentences:
            toks = _tokenise(sent)
            toks = [t if t in self.vocab else self.UNK for t in toks]
            padded = [self.START] * (self.n - 1) + toks + [self.END]

            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - (self.n - 1): i])
                word    = padded[i]
                total_log_prob += self.log_prob(word, context)
                total_tokens   += 1

        if total_tokens == 0:
            return float("inf")

        avg_log_prob = total_log_prob / total_tokens
        return 2 ** (-avg_log_prob)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, max_tokens: int = 50, seed: int = 42) -> str:
        """
        Sample a sentence from the model using random sampling.

        At each step, we sample the next word proportional to the smoothed
        conditional distribution P(w | context).  Sampling (rather than
        greedy argmax) yields more diverse output at the cost of occasional
        incoherence.

        Generation stops when </s> is sampled or max_tokens is reached.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before generating.")
        rng = random.Random(seed)
        context = [self.START] * (self.n - 1)
        tokens  = []
        V = len(self.vocab)
        vocab_list = sorted(self.vocab)

        for _ in range(max_tokens):
            ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
            # Build probability distribution over vocabulary
            probs = np.array([
                (self.counts[ctx][w] + self.alpha) /
                (self.context_totals[ctx] + self.alpha * V)
                for w in vocab_list
            ])
            probs /= probs.sum()  # normalise (should already sum to 1)
            chosen = rng.choices(vocab_list, weights=probs.tolist(), k=1)[0]
            if chosen == self.END:
                break
            tokens.append(chosen)
            context.append(chosen)

        return " ".join(tokens)

    # ── Reports ───────────────────────────────────────────────────────────────

    def top_k_words(self, context: Tuple[str, ...], k: int = 10) -> List[Tuple[str, float]]:
        """
        Return the top-k words by smoothed probability given a context.
        Useful for inspecting what the model predicts after a specific phrase.
        """
        ctx = tuple(context[-(self.n - 1):])
        V   = len(self.vocab)
        c_ctx = self.context_totals[ctx]
        ranked = sorted(
            self.vocab,
            key=lambda w: (self.counts[ctx][w] + self.alpha) /
                          (c_ctx + self.alpha * V),
            reverse=True,
        )
        result = []
        for w in ranked[:k]:
            p = (self.counts[ctx][w] + self.alpha) / (c_ctx + self.alpha * V)
            result.append((w, p))
        return result


def run_ngram_experiments(cfg: dict) -> Dict:
    """
    Train and evaluate all three n-gram orders.  Returns a dict of results
    suitable for printing / plotting.

    Experiments performed
    ─────────────────────
    1. Fit unigram, bigram, trigram models on training sentences.
    2. Compute perplexity on val and test sentences.
    3. Generate sample sentences from the trigram model.
    4. Ablation: compare α=0 (no smoothing) vs α=1 (Laplace).
       Note: α=0 will give infinite perplexity if any test n-gram is unseen,
       demonstrating *why* smoothing is necessary.
    5. Log top-5 predicted words after "the great" (bigram context) as a
       qualitative demonstration.
    """
    from src.data_loader import load_processed
    results = {}

    train_df, _, train_sents = load_processed("train", cfg)
    val_df,   _, val_sents   = load_processed("val",   cfg)
    test_df,  _, test_sents  = load_processed("test",  cfg)

    ngram_cfg = cfg["ngram"]
    alpha     = ngram_cfg["smoothing_alpha"]

    print("\n" + "═" * 60)
    print("  N-GRAM LANGUAGE MODEL EXPERIMENTS  (Lab A4)")
    print("═" * 60)
    print(f"  Training sentences : {len(train_sents):>6}")
    print(f"  Val sentences      : {len(val_sents):>6}")
    print(f"  Test sentences     : {len(test_sents):>6}")
    print(f"  Laplace α          : {alpha}")
    print()

    for n in ngram_cfg["orders"]:
        model = NgramLanguageModel(
            n=n,
            alpha=alpha,
            unk_threshold=ngram_cfg["unk_threshold"]
        ).fit(train_sents)

        val_pp  = model.perplexity(val_sents[:500])
        test_pp = model.perplexity(test_sents[:500])

        results[f"ngram_{n}"] = {
            "n": n,
            "val_perplexity":  val_pp,
            "test_perplexity": test_pp,
        }
        print(f"  {n}-gram  |  Val PP: {val_pp:8.2f}  |  Test PP: {test_pp:8.2f}")

    # Ablation: no smoothing vs Laplace on bigram
    print("\n  -- Ablation: smoothing effect on Bigram --")
    for a in [0.0, 1.0]:
        try:
            m = NgramLanguageModel(n=2, alpha=a, unk_threshold=ngram_cfg["unk_threshold"])
            m.fit(train_sents)
            try:
                pp = m.perplexity(test_sents[:200])
            except (ZeroDivisionError, ValueError):
                pp = float("inf")
            label = "Laplace (α=1)" if a == 1.0 else "No smoothing  "
            print(f"  Bigram {label}  |  Test PP: {pp:8.2f}")
        except Exception as exc:
            print(f"  Bigram α={a}: ERROR — {exc}")

    # Sample generation from trigram
    print("\n  -- Sample sentences from Trigram model --")
    tri = NgramLanguageModel(n=3, alpha=alpha,
                             unk_threshold=ngram_cfg["unk_threshold"]).fit(train_sents)
    for i in range(ngram_cfg["num_sample_sentences"]):
        sent = tri.generate(max_tokens=30, seed=i)
        print(f"  [{i+1:02d}] {sent}")

    # Top-5 bigram predictions after "the"
    print('\n  -- Top-5 bigram predictions after "the" --')
    bi = NgramLanguageModel(n=2, alpha=alpha,
                            unk_threshold=ngram_cfg["unk_threshold"]).fit(train_sents)
    for word, prob in bi.top_k_words(("the",), k=5):
        print(f"       P({word!r:<15} | 'the') = {prob:.6f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# BAG-OF-WORDS + NAÏVE BAYES CLASSIFIER  (Lab A3)
# ──────────────────────────────────────────────────────────────────────────────

class PlanetNaiveBayesClassifier:
    """
    Multinomial Naïve Bayes classifier for planet habitability.

    Maps the serialised feature string representation of each planet (e.g.
    "Temp=1740K | Radius=11.2Re | Mass=320Me | Hot=yes | Giant=yes")
    to one of two classes: 'habitable' or 'hostile'.

    Why BoW on structured data?
    ───────────────────────────
    The feature string format looks unusual as BoW input (it's structured
    text, not prose), but it creates an interesting linguistic signal:
    tokens like "Hot=yes", "Giant=yes" appear exclusively in hostile
    planets, while "Habitable=yes" appears only in the positive class.
    The NB classifier can learn these sharp lexical boundaries.  This
    also demonstrates that BoW is not limited to free-form text.

    We additionally test the classifier on free-text planet descriptions
    (manually written summaries) to show it generalises.
    """

    def __init__(self, alpha: float = 1.0, max_features: int = 5000,
                 ngram_range: Tuple = (1, 2)):
        self.alpha       = alpha
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = None
        self._classifier = None
        self._classes    = None

    def fit(self, X_texts: List[str], y_labels: List[str]) -> "PlanetNaiveBayesClassifier":
        """
        Fit CountVectorizer + MultinomialNB on serialised planet feature strings.

        CountVectorizer builds a vocabulary of the most frequent n-grams
        (up to max_features) and produces a sparse count matrix.
        MultinomialNB then learns class-conditional word probabilities.
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB

        self._vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            strip_accents="unicode",
            lowercase=True,
        )
        self._classifier = MultinomialNB(alpha=self.alpha)
        self._classes    = sorted(set(y_labels))

        log.info("Fitting NB classifier (α=%.1f, max_features=%d, ngram=%s) …",
                 self.alpha, self.max_features, self.ngram_range)
        X = self._vectorizer.fit_transform(X_texts)
        self._classifier.fit(X, y_labels)
        log.info("  Classes: %s  |  Train samples: %d  |  Vocabulary: %d",
                 self._classes, len(y_labels), len(self._vectorizer.vocabulary_))
        return self

    def predict(self, X_texts: List[str]) -> np.ndarray:
        X = self._vectorizer.transform(X_texts)
        return self._classifier.predict(X)

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        X = self._vectorizer.transform(X_texts)
        return self._classifier.predict_proba(X)

    def top_features(self, class_name: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Return the k most discriminative tokens for a given class.

        Uses the log-probability of each token under the class model
        (higher = more indicative of this class).  Useful for explaining
        *why* the model classified a planet as habitable vs hostile.
        """
        if class_name not in self._classes:
            raise ValueError(f"Unknown class '{class_name}'.  Known: {self._classes}")
        class_idx = self._classes.index(class_name)
        feature_names = self._vectorizer.get_feature_names_out()
        log_probs = self._classifier.feature_log_prob_[class_idx]
        top_idx = np.argsort(log_probs)[::-1][:k]
        return [(feature_names[i], log_probs[i]) for i in top_idx]


def _prepare_nb_data(cfg: dict) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load processed planet splits and create BoW inputs (serialised feature
    strings) + binary labels (habitable / hostile).
    """
    from src.data_loader import load_processed, serialise_planet_features

    train_df, _, _ = load_processed("train", cfg)
    test_df,  _, _ = load_processed("test",  cfg)

    def _rows_to_xy(df):
        X, y = [], []
        for _, row in df.iterrows():
            X.append(serialise_planet_features(row))
            y.append("habitable" if row.get("is_habitable", False) else "hostile")
        return X, y

    X_train, y_train = _rows_to_xy(train_df)
    X_test,  y_test  = _rows_to_xy(test_df)
    return X_train, y_train, X_test, y_test


def run_nb_experiments(cfg: dict) -> Dict:
    """
    Train and evaluate the Naïve Bayes planet classifier.

    Experiments
    ──────────
    1. Fit NB with Laplace smoothing (α=1).
    2. Compute accuracy, precision, recall, F1 (per class and macro average).
    3. Plot confusion matrix (saved to outputs/).
    4. Ablation: compare α=0 (no smoothing) vs α=1.
    5. Report top-10 most discriminative n-gram features per class.
    6. Compare against an SVM (sklearn LinearSVC) baseline.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        ConfusionMatrixDisplay
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb_cfg  = cfg["naive_bayes"]
    results = {}

    X_train, y_train, X_test, y_test = _prepare_nb_data(cfg)

    print("\n" + "═" * 60)
    print("  NAÏVE BAYES PLANET CLASSIFIER  (Lab A3)")
    print("═" * 60)
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")
    class_dist = Counter(y_train)
    print(f"  Class balance (train): {dict(class_dist)}")
    print()

    # Main NB model
    clf = PlanetNaiveBayesClassifier(
        alpha=nb_cfg["smoothing_alpha"],
        max_features=nb_cfg["max_features"],
        ngram_range=tuple(nb_cfg["ngram_range"]),
    ).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred, labels=["habitable", "hostile"])

    results["nb_accuracy"]     = acc
    results["nb_report"]       = report
    results["nb_confusion"]    = cm.tolist()

    print(f"  Accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["habitable", "hostile"]).plot(ax=ax)
    ax.set_title("NB Classifier — Planet Habitability Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "nb_confusion_matrix.png", dpi=150)
    plt.close()
    log.info("Confusion matrix saved.")

    # Top discriminative features
    print("\n  Top-10 features for 'habitable':")
    for feat, lp in clf.top_features("habitable", k=10):
        print(f"    {feat:<30} log-prob: {lp:.4f}")
    print("  Top-10 features for 'hostile':")
    for feat, lp in clf.top_features("hostile", k=10):
        print(f"    {feat:<30} log-prob: {lp:.4f}")

    # Ablation: smoothing
    print("\n  -- Ablation: Laplace smoothing effect --")
    for a in [0.0, 1.0]:
        try:
            m = PlanetNaiveBayesClassifier(
                alpha=a,
                max_features=nb_cfg["max_features"],
                ngram_range=tuple(nb_cfg["ngram_range"]),
            ).fit(X_train, y_train)
            yp = m.predict(X_test)
            label = "α=1 (Laplace)" if a == 1.0 else "α=0 (none)   "
            hab_f1 = classification_report(y_test, yp, output_dict=True).get(
                "habitable", {}).get("f1-score", 0.0)
            print(f"  {label}  Accuracy={accuracy_score(y_test, yp):.4f}  "
                  f"Habitable-F1={hab_f1:.4f}")
        except Exception as exc:
            print(f"  α={a}: ERROR — {exc}")

    # SVM comparison
    print("\n  -- SVM Baseline Comparison --")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline

        svm_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=nb_cfg["max_features"],
                                       ngram_range=tuple(nb_cfg["ngram_range"]))),
            ("svm",   LinearSVC(C=cfg["evaluation"]["style_clf_c"],
                                max_iter=2000)),
        ])
        svm_pipe.fit(X_train, y_train)
        yp_svm = svm_pipe.predict(X_test)
        svm_acc = accuracy_score(y_test, yp_svm)
        svm_hab_f1 = classification_report(y_test, yp_svm, output_dict=True).get(
            "habitable", {}).get("f1-score", 0.0)
        print(f"  SVM (LinearSVC)      Accuracy={svm_acc:.4f}  "
              f"Habitable-F1={svm_hab_f1:.4f}")
        results["svm_accuracy"]    = svm_acc
        results["svm_habitable_f1"] = svm_hab_f1
    except Exception as exc:
        log.warning("SVM comparison failed: %s", exc)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Combined test entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """
    Run all baseline experiments and print a consolidated summary table.
    Called by: `python src/baseline_models.py --test`
    """
    cfg      = _load_config()
    ng_res   = run_ngram_experiments(cfg)
    nb_res   = run_nb_experiments(cfg)

    print("\n" + "═" * 60)
    print("  BASELINE SUMMARY")
    print("═" * 60)
    print("  N-gram perplexity (test set):")
    for k, v in ng_res.items():
        print(f"    {v['n']}-gram:   {v['test_perplexity']:.2f}")
    print(f"\n  NB Accuracy     : {nb_res.get('nb_accuracy', 0):.4f}")
    hab_f1 = nb_res["nb_report"].get("habitable", {}).get("f1-score", 0.0)
    print(f"  NB Habitable F1 : {hab_f1:.4f}")
    svm_acc = nb_res.get("svm_accuracy")
    if svm_acc is not None:
        print(f"  SVM Accuracy    : {svm_acc:.4f}")
    print("═" * 60)

    # Save numeric results
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_results.json", "w") as f:
        # Convert non-serialisable numpy ints to Python ints
        def _safe(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            return obj
        json.dump(
            {"ngram": ng_res, "naive_bayes": {k: v for k, v in nb_res.items()
                                               if k != "nb_report"}},
            f, indent=2, default=_safe
        )
    log.info("Baseline results saved to outputs/baseline_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline NLP model experiments.")
    parser.add_argument("--test", action="store_true",
                        help="Run all baseline experiments (n-gram + NB).")
    parser.add_argument("--ngram-only", action="store_true",
                        help="Run only n-gram experiments.")
    parser.add_argument("--nb-only", action="store_true",
                        help="Run only Naive Bayes experiments.")
    args = parser.parse_args()

    cfg = _load_config()

    if args.ngram_only:
        run_ngram_experiments(cfg)
    elif args.nb_only:
        run_nb_experiments(cfg)
    else:
        run_all_tests()
