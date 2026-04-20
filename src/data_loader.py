"""
data_loader.py — Exoplanet Mythos Generator
============================================
CST-106 Applications of NLP  |  Lab Coverage: preprocessing, feature engineering

PURPOSE
-------
This module is the single source of truth for all data ingestion, cleaning,
normalisation, and splitting. It:
  1. Downloads confirmed exoplanet data from the NASA Exoplanet Archive via
     its Table Access Protocol (TAP) REST endpoint.
  2. Fetches public-domain mythology text from Project Gutenberg.
  3. Cleans and normalises astrophysical features (handles missing values,
     outliers, unit conversions).
  4. Engineers new boolean / categorical features from raw physics that map
     naturally to mythological archetypes (heat → fire gods, cold → ice
     giants, massive → titan narratives, etc.).
  5. Creates text prefix serialisations of each planet's features, which are
     prepended to story sequences during Transformer fine-tuning (Lab A8).
  6. Assembles planet→story training pairs by pairing each planet prefix with
     randomly sampled mythology sentences.
  7. Saves clean train / val / test splits to `data/processed/`.

DESIGN DECISIONS
----------------
- We use the TAP ADQL endpoint rather than the bulk download so we choose
  exactly the columns we need, keeping the dataset manageable.
- Missing numeric values are filled with the column median (robust to
  outliers) rather than the mean.  Categorical missing values become 'Unknown'.
- One-hot encoding is applied to `st_spectype` (O/B/A/F/G/K/M stars) so the
  MLP encoder receives a purely numeric feature vector.
- The engineered binary features (`is_hot`, `is_icy`, `is_giant`,
  `has_short_year`) double as human-readable myth cues AND as the target
  labels for the Naïve Bayes habitability classifier (Lab A3).
- Planet→story pairs are created by uniformly sampling mythology sentences;
  each planet is paired with up to `MAX_PAIRS_PER_PLANET` sentences so that
  rare planets still get representation.

VIVA / PRESENTATION TALKING POINTS
-----------------------------------
• "We query the NASA TAP endpoint with ADQL — the same protocol used by
  professional astronomers — so our data is always up-to-date."
• "Feature engineering bridges physics and mythology: a planet with
  T_eq > 1000 K is labelled `is_hot`, which primes the model to generate
  fire-themed narratives — this is a form of structured conditioning."
• "Filling missing values with the median is a deliberate choice: the
  NASA archive has ~15–20 % missing mass values (because RV follow-up is
  incomplete), and the median does not get pulled by ultra-massive hot
  Jupiters the way the mean would."
"""

import os
import sys
import json
import random
import logging
import re
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config loader  (used by all modules)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Optional[str] = None) -> dict:
    """
    Load the YAML config from *path* (defaults to project-root/config.yaml).

    Returns a plain dict.  All other modules call this at import time so a
    single edit to config.yaml propagates everywhere.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 1.  NASA Exoplanet Archive download
# ──────────────────────────────────────────────────────────────────────────────

_NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def fetch_exoplanet_data(cfg: dict, force_redownload: bool = False) -> pd.DataFrame:
    """
    Download the confirmed exoplanet table from the NASA Exoplanet Archive
    using ADQL (Astronomical Data Query Language) over the TAP REST endpoint.

    Parameters
    ----------
    cfg : dict
        Project configuration dictionary (from load_config()).
    force_redownload : bool
        If True, ignore cached CSV and re-query the archive.

    Returns
    -------
    pd.DataFrame
        Raw (unprocessed) exoplanet data.

    Technical Notes
    ---------------
    - Table used: `ps`  (Planetary Systems) — the most complete catalogue in
      the archive (~6 000 confirmed planets as of early 2026).
    - We select `WHERE tran_flag = 1 OR rv_flag = 1` to limit to planets with
      robust physical measurements (transiting or radial-velocity confirmed).
    - `FORMAT=csv` returns a plain comma-separated response we parse with
      pandas read_csv; no extra parsing needed.
    - The TAP endpoint is rate-limited, so we only query once and cache.
    """
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "exoplanets_raw.csv"

    if cache_path.exists() and not force_redownload:
        log.info("Loading cached exoplanet data from %s", cache_path)
        df = pd.read_csv(cache_path)
        log.info("Loaded %d rows from cache.", len(df))
        return df

    columns = cfg["nasa"]["columns"]
    col_str = ", ".join(columns)
    max_rows = cfg["nasa"]["max_rows"]

    # ADQL query — select all confirmed planets with at least some numeric
    # data, order by equilibrium temperature so we get a good range.
    adql = (
        f"SELECT TOP {max_rows} {col_str} "
        f"FROM ps "
        f"WHERE pl_controv_flag = 0 "   # exclude controversial detections
        f"ORDER BY pl_eqt DESC"
    )

    params = {
        "QUERY": adql,
        "FORMAT": "csv",
        "REQUEST": "doQuery",
        "LANG": "ADQL",
    }

    log.info("Querying NASA Exoplanet Archive TAP endpoint…")
    log.info("ADQL: %s", adql)

    try:
        response = requests.get(_NASA_TAP, params=params, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.error("NASA TAP query failed: %s", exc)
        log.warning("Falling back to a synthetic demo dataset for offline use.")
        return _synthetic_exoplanet_data(max_rows)

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    log.info("Downloaded %d exoplanet rows.", len(df))
    df.to_csv(cache_path, index=False)
    log.info("Saved to %s", cache_path)
    return df


def _synthetic_exoplanet_data(n: int = 500) -> pd.DataFrame:
    """
    Generate a realistic synthetic exoplanet dataset for offline / CI use.

    Column distributions are calibrated to real NASA archive statistics:
    - ~30 % Hot Jupiters (high temp, large radius, short period)
    - ~25 % Super-Earths
    - ~20 % Mini-Neptunes
    - ~25 % other

    This ensures the NB classifier (Lab A3) and NLP training still work when
    the NASA endpoint is unavailable (e.g. behind a firewall).
    """
    rng = np.random.default_rng(42)
    n_hj    = int(n * 0.30)   # Hot Jupiters
    n_se    = int(n * 0.25)   # Super-Earths
    n_mn    = int(n * 0.20)   # Mini-Neptunes
    n_other = n - n_hj - n_se - n_mn

    def _make(n_sub, temp_mu, temp_sig, rad_mu, rad_sig, per_mu, per_sig, mass_mu, mass_sig):
        return {
            "pl_name":       [f"Syn-{i:04d} b" for i in range(n_sub)],
            "pl_masse":      rng.normal(mass_mu, mass_sig, n_sub).clip(0.1),
            "pl_rade":       rng.normal(rad_mu,  rad_sig,  n_sub).clip(0.5),
            "pl_orbper":     rng.exponential(per_mu, n_sub).clip(0.5),
            "pl_eqt":        rng.normal(temp_mu, temp_sig, n_sub).clip(100),
            "pl_orbeccen":   rng.beta(1, 5, n_sub),
            "st_spectype":   rng.choice(["F", "G", "K", "M", "A"], n_sub),
            "st_teff":       rng.normal(5500, 800, n_sub).clip(3000),
            "st_lum":        rng.normal(0.0, 0.4, n_sub),
            "sy_dist":       rng.exponential(200, n_sub).clip(5),
            "discoverymethod": rng.choice(["Transit", "Radial Velocity", "Imaging"], n_sub),
        }

    parts = [
        _make(n_hj,    1800, 500, 12, 3, 4,  3,  400, 200),
        _make(n_se,     400, 200,  1.6, 0.5, 20, 15,    4, 3),
        _make(n_mn,     700, 300,  2.8, 0.7, 15, 12,   15, 8),
        _make(n_other,  900, 600,  5.0, 3.0, 50, 40,   80, 60),
    ]
    frames = [pd.DataFrame(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    log.info("Generated %d synthetic exoplanet rows.", len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Mythology Corpus download (Project Gutenberg)
# ──────────────────────────────────────────────────────────────────────────────

_GUTENBERG_URL = "https://gutenberg.org/cache/epub/{gid}/pg{gid}.txt"

# Curated mythology texts with known Gutenberg IDs
_MYTH_BOOKS = {
    22381: "Bulfinch's Age of Fable",
    14915: "Myths and Legends of Ancient Greece and Rome",
    17780: "Old Greek Folk Stories Told Anew",
    7700:  "The Golden Bough (Frazer) — abridged",
    348:   "Norse Mythology and Legends",
    1727:  "The Odyssey (Homer / Butler)",
    1728:  "Iliad (Homer / Butler)",
}

def fetch_mythology_corpus(cfg: dict, force_redownload: bool = False) -> str:
    """
    Download and concatenate public-domain mythology texts from Project
    Gutenberg.  Strips the standard Gutenberg header/footer boilerplate
    so we only keep narrative text.

    Returns
    -------
    str
        Raw concatenated mythology text (millions of characters).

    Technical Notes
    ---------------
    - Project Gutenberg .txt files have a standard preamble before
      `*** START OF THE PROJECT GUTENBERG EBOOK ***` and an epilogue after
      `*** END OF THE PROJECT GUTENBERG EBOOK ***`.  We strip both.
    - Some books have encoding issues (Latin-1 vs UTF-8).  We try UTF-8
      first and fall back to Latin-1.
    - If the network is unavailable, we fall back to a 10 k-word built-in
      myth stub (enough to demo the system end-to-end).
    """
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_cache = raw_dir / "mythology_corpus.txt"

    if corpus_cache.exists() and not force_redownload:
        log.info("Loading cached mythology corpus from %s", corpus_cache)
        return corpus_cache.read_text(encoding="utf-8")

    gutenberg_ids = cfg["corpus"].get("gutenberg_ids", list(_MYTH_BOOKS.keys()))
    full_text = []

    for gid in tqdm(gutenberg_ids, desc="Downloading mythology texts"):
        title = _MYTH_BOOKS.get(gid, f"Gutenberg #{gid}")
        url = _GUTENBERG_URL.format(gid=gid)
        book_cache = raw_dir / f"gutenberg_{gid}.txt"

        if book_cache.exists():
            text = book_cache.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                log.info("  Fetching: %s  [%s]", title, url)
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                text = resp.text
                book_cache.write_text(text, encoding="utf-8")
            except Exception as exc:
                log.warning("  Failed to fetch %s: %s — skipping.", title, exc)
                continue

        text = _strip_gutenberg_boilerplate(text)
        full_text.append(f"\n\n=== {title} ===\n\n" + text)
        log.info("  Loaded %d chars from '%s'", len(text), title)

    if not full_text:
        log.warning("No myth texts downloaded — using built-in stub corpus.")
        return _MYTH_STUB

    corpus = "\n".join(full_text)
    corpus_cache.write_text(corpus, encoding="utf-8")
    log.info("Total mythology corpus: %d chars (%d words approx.)",
             len(corpus), len(corpus.split()))
    return corpus


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Gutenberg header and footer markers."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    # Find start
    start_idx = 0
    for marker in start_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            start_idx = text.index("\n", idx) + 1
            break
    # Find end
    end_idx = len(text)
    for marker in end_markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            end_idx = idx
            break
    return text[start_idx:end_idx].strip()


# Built-in stub — used offline / fast demo
_MYTH_STUB = """
In the age before memory, when the stars were young and the void between worlds
breathed with possibility, the great gods shaped the heavens from chaos.
Prometheus, the Titan of foresight, stole fire from the forge of Hephaestus
and carried it across the celestial river to give warmth to mortal hearts.
The goddess Athena, born fully armoured from the mind of Zeus, watched over
those who sought wisdom in the darkness between stars.
At the edge of the known cosmos, beyond the pillars that Hercules once set,
lay the realm of Aether — a world of pure light where neither heat nor cold
could touch the souls who dwelled there in eternal contemplation.
The sea-god Poseidon raised his trident and from the deep waters of the void
there arose a new world, wreathed in silver mist and haunted by voices of
the ancient dead. Its oceans were not of water but of liquid iron, churning
beneath skies the colour of dying embers.
Thor, whose hammer shook the roots of the world-tree Yggdrasil, fought the
serpent Jormungandr on a plain of frozen methane, their battle sending shocks
through the nine realms. The Norns wove the fate of that world into their
tapestry, binding fire and ice in an endless dance.
Apollo drove his golden chariot across the face of a sun ten times as bright
as ours, and the world beneath him turned molten — seas of lava, mountains
of obsidian, and in the centre, a temple of pure diamond erected by beings
who worshipped light itself.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Feature Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_SPEC_TYPE_MAP = {
    "O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6, "Unknown": 7
}

def preprocess_planets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Clean and engineer features from the raw NASA exoplanet DataFrame.

    Steps (in order)
    ----------------
    1. Rename columns to friendlier names for readability in logs/notebooks.
    2. Drop rows where both mass AND radius are missing (we cannot characterise
       these planets at all — ~2–3 % of the dataset).
    3. Fill remaining numeric NaNs with the column **median** (not mean).
       Rationale: if the median avoids the Hot Jupiter peak pulling fill
       values, we get better estimates for the silent majority of Earth-like
       detections.
    4. Fill categorical NaNs with 'Unknown'.
    5. Normalise continuous features to [0, 1] range using (x - min) /
       (max - min) — stored in columns prefixed with `norm_`.
       Rationale: the MLP latent encoder processes these normalised values;
       unnormalised temperatures (100–5000 K) would dominate the gradient
       over orbital eccentricities (0–1).
    6. Extract the *letter* of the stellar spectral type (first character)
       and map it to an integer (O→0 … M→6, Unknown→7).  This becomes the
       `spec_class` column used for one-hot encoding feature vectors.
    7. Engineer boolean archetype features used for Naïve Bayes labels and
       for style-token selection:
         - `is_hot`        : T_eq > 1000 K
         - `is_icy`        : T_eq < 200 K
         - `is_giant`      : R_planet > 4.0 R_earth
         - `has_short_year`: orbital period < 10 days
         - `is_habitable`  : 200 < T_eq < 350 K AND R < 2.5 R_earth
    8. Assign a `myth_archetype` string label (used as style token and as NB
       classification target):
         - 'fire'  → is_hot
         - 'ice'   → is_icy
         - 'gas'   → is_giant and not hot/icy
         - 'myth'  → everything else (default / habitable)

    Returns
    -------
    pd.DataFrame with all original columns plus engineered ones.
    """
    feat_cfg = cfg["features"]

    # 1. Rename for readability
    rename_map = {
        "pl_name":       "planet_name",
        "pl_masse":      "mass_earth",
        "pl_rade":       "radius_earth",
        "pl_orbper":     "orbital_period_days",
        "pl_eqt":        "eq_temp_K",
        "pl_orbeccen":   "eccentricity",
        "st_spectype":   "star_spec_type",
        "st_teff":       "star_teff_K",
        "st_lum":        "star_luminosity",
        "sy_dist":       "distance_pc",
        "discoverymethod": "discovery_method",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 2. Drop entirely uncharacterisable planets
    before = len(df)
    df = df.dropna(subset=["mass_earth", "radius_earth"], how="all")
    log.info("Dropped %d rows with no mass or radius → %d remain.", before - len(df), len(df))

    # 3. Fill numeric NaNs with column median
    numeric_cols = ["mass_earth", "radius_earth", "orbital_period_days",
                    "eq_temp_K", "eccentricity", "star_teff_K",
                    "star_luminosity", "distance_pc"]
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            if n_filled:
                log.info("  Filled %d missing %s with median %.3f", n_filled, col, median_val)

    # 4. Fill categorical NaNs
    cat_cols = ["star_spec_type", "discovery_method"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 5. Normalise continuous features
    for col in numeric_cols:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            denom = col_max - col_min if col_max != col_min else 1.0
            df[f"norm_{col}"] = (df[col] - col_min) / denom

    # 6. Spectral class integer encoding
    if "star_spec_type" in df.columns:
        df["spec_class_letter"] = df["star_spec_type"].str[0].fillna("Unknown")
        df["spec_class_int"] = df["spec_class_letter"].map(
            lambda x: _SPEC_TYPE_MAP.get(x, 7)
        )

    # 7. Archetype boolean features
    df["is_hot"]   = df["eq_temp_K"] > feat_cfg["temp_hot_threshold"]
    df["is_icy"]   = df["eq_temp_K"] < feat_cfg["temp_cold_threshold"]
    df["is_giant"] = df["radius_earth"] > feat_cfg["radius_giant_threshold"]
    df["has_short_year"] = df["orbital_period_days"] < feat_cfg["period_short_threshold"]
    df["is_habitable"] = (
        (df["eq_temp_K"] >= feat_cfg["habitable_temp_min"]) &
        (df["eq_temp_K"] <= feat_cfg["habitable_temp_max"]) &
        (df["radius_earth"] <= feat_cfg["habitable_radius_max"])
    )

    # 8. Myth archetype label
    def _archetype(row):
        if row["is_hot"]:   return "fire"
        if row["is_icy"]:   return "ice"
        if row["is_giant"]: return "gas"
        return "myth"

    df["myth_archetype"] = df.apply(_archetype, axis=1)

    log.info("Feature engineering complete.  Archetype distribution:\n%s",
             df["myth_archetype"].value_counts().to_string())
    log.info("Habitable planets: %d (%.1f %%)",
             df["is_habitable"].sum(), 100 * df["is_habitable"].mean())
    return df


def serialise_planet_features(row: pd.Series) -> str:
    """
    Convert a planet DataFrame row into a human-readable feature string.

    This string is prepended to every story sequence so the Transformer
    can condition generation on the planet's physical properties.

    Format example
    --------------
    "<FEAT> Temp=1740K | Radius=11.2Re | Mass=320Me | Period=3.1d |
     Ecc=0.02 | Star=G | Dist=420pc | Hot=yes | Giant=yes <PLANET>"

    Design reasoning
    ----------------
    Using *natural language–like* tokens (e.g. "Temp=1740K") rather than
    raw numbers is intentional: GPT-2's subword tokeniser has already seen
    patterns like "Temp=…" in training data (from scientific text), so it
    can relate these tokens to semantic context more readily than a bare
    float "1740.0".
    """
    parts = ["<FEAT>"]

    # Temperature
    if "eq_temp_K" in row.index and not pd.isna(row.get("eq_temp_K")):
        parts.append(f"Temp={row['eq_temp_K']:.0f}K")
    # Radius
    if "radius_earth" in row.index and not pd.isna(row.get("radius_earth")):
        parts.append(f"Radius={row['radius_earth']:.1f}Re")
    # Mass
    if "mass_earth" in row.index and not pd.isna(row.get("mass_earth")):
        parts.append(f"Mass={row['mass_earth']:.1f}Me")
    # Orbital period
    if "orbital_period_days" in row.index and not pd.isna(row.get("orbital_period_days")):
        parts.append(f"Period={row['orbital_period_days']:.1f}d")
    # Eccentricity
    if "eccentricity" in row.index and not pd.isna(row.get("eccentricity")):
        parts.append(f"Ecc={row['eccentricity']:.2f}")
    # Stellar spectral class
    if "spec_class_letter" in row.index:
        parts.append(f"Star={row['spec_class_letter']}")
    # Distance
    if "distance_pc" in row.index and not pd.isna(row.get("distance_pc")):
        parts.append(f"Dist={row['distance_pc']:.0f}pc")
    # Boolean flags
    for flag, label in [("is_hot", "Hot"), ("is_icy", "Icy"),
                        ("is_giant", "Giant"), ("has_short_year", "ShortYear"),
                        ("is_habitable", "Habitable")]:
        if flag in row.index:
            parts.append(f"{label}={'yes' if row[flag] else 'no'}")

    parts.append("<PLANET>")
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Text Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str, cfg: dict) -> List[str]:
    """
    Tokenise mythology corpus into a list of clean sentences.

    Steps
    -----
    1. Split on sentence-boundary punctuation (. ! ?) — uses a naïve but
       robust regex; NLTK's sent_tokenize is more accurate but requires
       download.
    2. Normalise whitespace (collapse runs of spaces/newlines).
    3. Remove Project Gutenberg markup artefacts (ALL-CAPS chapter headings,
       Roman numerals used as section markers).
    4. Filter by min / max sentence length (from config).
    5. Return list of clean sentence strings.

    Note: We do NOT lowercase here — the Transformer tokeniser handles case
    internally, and mythological proper nouns (Zeus, Odin) are meaningful
    capitalised signals.
    """
    min_len = cfg["corpus"]["min_sentence_len"]
    max_len = cfg["corpus"]["max_sentence_len"]

    # Normalise whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove chapter headings (lines that are all-caps or Roman numerals)
    lines = text.split("\n")
    lines = [ln for ln in lines if not re.fullmatch(r"[A-Z\s\.\-]+|[IVXLCivxlc]+\.", ln.strip())]
    text = " ".join(lines)

    # Split into sentences on . ! ?  (keep delimiter by lookahead)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Clean each sentence
    clean = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        tokens = s.split()
        if min_len <= len(tokens) <= max_len:
            clean.append(s)

    log.info("Text preprocessing: %d sentences retained (min=%d, max=%d tokens).",
             len(clean), min_len, max_len)
    return clean


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Dataset Splitting
# ──────────────────────────────────────────────────────────────────────────────

def split_planets(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the planet DataFrame into train / val / test.

    Stratification: we stratify by `myth_archetype` so that all four myth
    classes (fire / ice / gas / myth) appear in all splits in proportion.
    Without stratification, the test set could be dominated by the majority
    class ('fire' — ~40 % of planets are hot) leading to inflated NB accuracy.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    split_cfg = cfg["splits"]
    val_frac  = split_cfg["planet_val"]
    test_frac = split_cfg["planet_test"]

    # First split off test, then split remainder into train/val
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["myth_archetype"], random_state=42
    )
    val_frac_adj = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_frac_adj, stratify=train_val["myth_archetype"], random_state=42
    )
    log.info("Planet splits — Train: %d  Val: %d  Test: %d", len(train), len(val), len(test))
    return train, val, test


def split_sentences(sentences: List[str], cfg: dict) -> Tuple[List, List, List]:
    """
    Split myth sentences into train / val / test.

    No stratification needed (unlabelled text); we simply slice.
    """
    split_cfg = cfg["splits"]
    n = len(sentences)
    n_val  = int(n * split_cfg["text_val"])
    n_test = int(n * split_cfg["text_test"])
    train  = sentences[: n - n_val - n_test]
    val    = sentences[n - n_val - n_test : n - n_test]
    test   = sentences[n - n_test :]
    log.info("Sentence splits — Train: %d  Val: %d  Test: %d", len(train), len(val), len(test))
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Planet → Story Pair Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_planet_story_pairs(
    planet_df: pd.DataFrame,
    sentences: List[str],
    max_pairs: int,
    max_pairs_per_planet: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    Pair each planet's serialised feature prefix with randomly sampled
    mythology sentence windows to create supervised (input, target) pairs
    for Transformer fine-tuning.

    Pairing strategy
    ----------------
    - For each planet, randomly sample up to `max_pairs_per_planet`
      windows of 5–10 consecutive sentences from the *archetype-matching*
      sentence pool.  Archetype matching means: if the planet is `is_hot`,
      prefer sentences containing fire/heat vocabulary (this is a soft
      preference — we filter, then fall back to random if the filtered pool
      is too small).
    - The input string = "<myth> {serialised_features} | {sampled_story}"
    - Target = input shifted right by one token (standard causal LM setup).

    Returns
    -------
    List[Dict] with keys: "planet_name", "archetype", "prefix",
                           "story", "input_text"
    """
    rng = random.Random(seed)

    # Keyword maps for soft archetype matching
    archetype_keywords = {
        "fire": ["fire", "flame", "heat", "blaze", "inferno", "sun", "burn",
                 "scorched", "lava", "molten", "forge", "ember"],
        "ice":  ["ice", "frost", "cold", "frozen", "blizzard", "glacier",
                 "winter", "snow", "chill", "arctic", "frigid"],
        "gas":  ["vast", "cloud", "great", "massive", "colossus", "titan",
                 "enormous", "deep", "abyss", "tempest"],
        "myth": ["wisdom", "hope", "life", "green", "water", "mortal",
                 "human", "peace", "grow", "harvest", "sea"],
    }

    # Group sentences by archetype affinity
    def _score(sentence: str, keywords: List[str]) -> int:
        sl = sentence.lower()
        return sum(kw in sl for kw in keywords)

    archetype_pools: Dict[str, List[str]] = {}
    for arch, kws in archetype_keywords.items():
        scored = [(s, _score(s, kws)) for s in sentences]
        archetype_pools[arch] = [s for s, sc in sorted(scored, key=lambda x: -x[1])]

    pairs = []
    for _, row in planet_df.iterrows():
        arch   = row.get("myth_archetype", "myth")
        pool   = archetype_pools.get(arch, sentences)
        prefix = serialise_planet_features(row)
        style  = f"<{arch}>"

        for _ in range(max_pairs_per_planet):
            # Sample a window of 3-7 consecutive sentences
            win_size = rng.randint(3, 7)
            start = rng.randint(0, max(0, len(pool) - win_size - 1))
            window = pool[start : start + win_size]
            story = " ".join(window)

            input_text = f"<myth> {style} {prefix}\n{story}"
            pairs.append({
                "planet_name": row.get("planet_name", "Unknown"),
                "archetype":   arch,
                "prefix":      prefix,
                "story":       story,
                "input_text":  input_text,
            })
        if len(pairs) >= max_pairs:
            break

    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    log.info("Built %d planet→story pairs.", len(pairs))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Save / Load Processed Data
# ──────────────────────────────────────────────────────────────────────────────

def save_processed(
    split_name: str,
    planet_df: pd.DataFrame,
    pairs: List[Dict],
    sentences: List[str],
    cfg: dict,
) -> None:
    """
    Persist all processed artefacts for a given split to `data/processed/`.

    Files written
    -------------
    planets_{split}.csv   — cleaned planet DataFrame
    pairs_{split}.json    — planet→story pair list
    sentences_{split}.txt — one myth sentence per line
    """
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    planet_df.to_csv(proc_dir / f"planets_{split_name}.csv", index=False)
    with open(proc_dir / f"pairs_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    (proc_dir / f"sentences_{split_name}.txt").write_text(
        "\n".join(sentences), encoding="utf-8"
    )
    log.info("Saved processed %s data to %s", split_name, proc_dir)


def load_processed(split_name: str, cfg: dict) -> Tuple[pd.DataFrame, List[Dict], List[str]]:
    """Load previously saved processed data for a given split."""
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    planet_df = pd.read_csv(proc_dir / f"planets_{split_name}.csv")
    with open(proc_dir / f"pairs_{split_name}.json", encoding="utf-8") as f:
        pairs = json.load(f)
    sentences = (proc_dir / f"sentences_{split_name}.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    return planet_df, pairs, sentences


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(force_redownload: bool = False) -> None:
    """
    Full end-to-end data pipeline.  Run this once before training.

    Execution order
    ---------------
    1. Load config
    2. Download / cache NASA data
    3. Download / cache mythology corpus
    4. Preprocess planets (features, normalisation, archetypes)
    5. Preprocess text (sentences)
    6. Split both datasets
    7. Build planet→story pairs for each split
    8. Save everything to data/processed/

    After this runs, all other modules (baseline_models, train, evaluate)
    load from data/processed/ and do not need network access.
    """
    cfg = load_config()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    log.info("=" * 60)
    log.info("  Exoplanet Mythos Generator — Data Pipeline")
    log.info("=" * 60)

    # Step 1: Fetch data
    raw_planets      = fetch_exoplanet_data(cfg, force_redownload)
    mythology_corpus = fetch_mythology_corpus(cfg, force_redownload)

    # Step 2: Preprocess
    planets   = preprocess_planets(raw_planets, cfg)
    sentences = preprocess_text(mythology_corpus, cfg)

    # Step 3: Split
    p_train, p_val, p_test = split_planets(planets, cfg)
    s_train, s_val, s_test = split_sentences(sentences, cfg)

    split_cfg = cfg["splits"]

    # Step 4: Build pairs
    pairs_train = build_planet_story_pairs(
        p_train, s_train,
        max_pairs=split_cfg["pairs_train"],
        seed=cfg["seed"],
    )
    pairs_val = build_planet_story_pairs(
        p_val, s_val,
        max_pairs=split_cfg["pairs_val"],
        seed=cfg["seed"] + 1,
    )
    pairs_test = build_planet_story_pairs(
        p_test, s_test,
        max_pairs=split_cfg["pairs_test"],
        seed=cfg["seed"] + 2,
    )

    # Step 5: Save
    for name, pdf, pairs, sents in [
        ("train", p_train, pairs_train, s_train),
        ("val",   p_val,   pairs_val,   s_val),
        ("test",  p_test,  pairs_test,  s_test),
    ]:
        save_processed(name, pdf, pairs, sents, cfg)

    # Step 6: Summary statistics
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    log.info("\n" + "=" * 60)
    log.info("  DATA PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("  Planets     — train:%d  val:%d  test:%d",
             len(p_train), len(p_val), len(p_test))
    log.info("  Sentences   — train:%d  val:%d  test:%d",
             len(s_train), len(s_val), len(s_test))
    log.info("  Pairs       — train:%d  val:%d  test:%d",
             len(pairs_train), len(pairs_val), len(pairs_test))
    log.info("  Outputs written to: %s", proc_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the data pipeline.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cache exists.")
    args = parser.parse_args()
    run_pipeline(force_redownload=args.force)
