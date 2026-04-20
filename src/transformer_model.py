"""
transformer_model.py — Exoplanet Mythos Generator
==================================================
CST-106 Applications of NLP
Lab Coverage:
  • Lab A7 — Transformer Embedding + Self-Attention
  • Lab A8 — Fine-tuning Pretrained Transformer (GPT-2)

PURPOSE
───────
This module defines the full Transformer-based conditional language model
that generates mythological narratives conditioned on exoplanet features.

Architecture Overview
──────────────────────

  [Planet Features] → [MLP Latent Encoder] → [Prefix Embedding Adapter]
                                                         │
  [Story Prefix Text] → [GPT-2 Tokeniser] → [Token Embeddings]
                                                         │
                              ┌──────────────────────────┘
                              ▼
                   [GPT-2 Transformer Blocks]
                   (12 layers, 12 heads, d=768 for gpt2-medium)
                   Each block: LayerNorm → MHA → Add → LayerNorm → FFN → Add
                              │
                              ▼
                   [LM Head (Linear projection)] → [Vocabulary logits]
                              │
                              ▼
                   [Generated myth story tokens]

Key Design Decisions
─────────────────────
1. **Feature Prefix Strategy** (chosen over full latent injection):
   Planet features are serialised as a text prefix string and tokenised
   alongside the story text.  Example input to the model:
     "<myth> <fire> <FEAT> Temp=1740K | Radius=11.2Re … <PLANET>
      In the age of burning skies, where stars bled crimson…"
   The prefix approach is interpretable, requires no architectural changes
   to GPT-2 (works with HuggingFace out-of-the-box), and benefits from
   GPT-2's pre-trained understanding of temperature/radius tokens.

2. **MLP Latent Encoder** (optional, additive to first layer):
   A 2-layer MLP maps the normalised planet feature vector (20 dims) to a
   128-dim latent vector.  This vector is added to the first token
   embeddings of the prefix, giving the model a continuous numeric signal
   in addition to the discrete text prefix.  This demonstrates the
   "latent space conditioning" concept from the project spec.

3. **Special Tokens**:
   We add `<myth>`, `<fire>`, `<water>`, `<ice>`, `<gas>`, `<PLANET>`,
   `<FEAT>`, `<PAD>` to GPT-2's vocabulary.  These serve as:
   - Style control tokens (switch narrative register)
   - Structural delimiters (separate features from story text)
   The model learns embeddings for these tokens during fine-tuning.

4. **Causal Language Modelling**:
   The model is trained with a standard next-token prediction objective.
   The loss is computed only on the *story* portion (tokens after `<PLANET>`),
   not on the feature prefix — this ensures the model generates the story,
   not the features.  This is implemented by masking prefix token labels to
   -100 (ignored by PyTorch CrossEntropyLoss).

SELF-ATTENTION INTERNALS (Lab A7)
──────────────────────────────────
GPT-2 uses multi-head causal self-attention:
  Attention(Q, K, V) = softmax(QK^T / √d_k) V

For n=12 heads, hidden size 768:
  d_k = 768 / 12 = 64 per head
  Each head learns a different (Q, K, V) projection.

During generation, we extract the attention weights
(output_attentions=True) and visualise them in the notebook.
Research (Vig & Belinkov 2019, Clark et al. 2019) shows different heads
specialise:
  - Some attend to preceding nouns (syntactic agreement)
  - Some attend to the beginning of sentence
  - Some attend to the feature prefix tokens (our hypothesis)

We will verify this last point by checking if heads 0–2 (early layers)
attend heavily to the <FEAT> … <PLANET> prefix region.

VIVA / PRESENTATION TALKING POINTS
────────────────────────────────────
• "GPT-2 is a 12-layer, 12-head, 768-dimensional autoregressive Transformer.
  It was pre-trained on WebText (8 million web pages).  We fine-tune it on
  mythology text so it learns to generate myth-style prose."
• "Adding special tokens and fine-tuning for 3–5 epochs teaches the model to
  condition generation on planetary features in the prefix.  The model never
  saw `<FEAT> Temp=1740K` during pre-training, so the new embeddings are
  randomly initialised and trained from scratch."
• "The MLP latent encoder ensures continuous-valued physics (temperature,
  mass) influence the first-layer representation even before attention acts
  on the prefix text.  This is the 'latent space conditioning' concept."
• "We mask prefix token labels to -100 so the loss is not wasted on
  predicting the feature string (which is always provided as input) — only
  the story generation is rewarded."
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Config helper
# ──────────────────────────────────────────────────────────────────────────────
def _load_config():
    from src.data_loader import load_config
    return load_config()


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Tokeniser factory
# ──────────────────────────────────────────────────────────────────────────────

def build_tokeniser(cfg: dict):
    """
    Load the GPT-2 BPE tokeniser and add project-specific special tokens.

    HuggingFace's `AutoTokenizer` handles the BPE vocabulary and fast
    tokenisation.  We explicitly set the padding token to <PAD> (GPT-2
    has no pad token by default) and resize the model's token embedding
    table after adding new tokens.

    Returns
    -------
    tokenizer : PreTrainedTokenizerFast
    """
    from transformers import AutoTokenizer

    model_name  = cfg["transformer"]["model_name"]
    spec_tokens = cfg["transformer"]["special_tokens"]

    log.info("Loading tokeniser: %s", model_name)
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 has no pad token — we reuse <PAD> from our special tokens
    tokeniser.add_special_tokens({
        "pad_token":              "<PAD>",
        "additional_special_tokens": [t for t in spec_tokens if t != "<PAD>"],
    })

    log.info("Tokeniser vocabulary size after adding special tokens: %d",
             len(tokeniser))
    return tokeniser


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MLP Latent Encoder
# ──────────────────────────────────────────────────────────────────────────────

class PlanetLatentEncoder(nn.Module):
    """
    A 2-layer MLP that maps a normalised planet feature vector to a latent
    representation of size `latent_dim`.

    Architecture
    ────────────
      Input  (input_dim)  → Linear → GELU → Dropout(0.1)
      Hidden (hidden_dim) → Linear → GELU → Dropout(0.1)
      Output (latent_dim)

    The output vector is added to the first `latent_dim` dimensions of the
    token embedding at position 0 of the feature prefix.

    Using GELU activations matches GPT-2's internal activation function,
    making the representations more compatible.

    Parameters
    ──────────
    input_dim  : int — number of raw numeric planet features (~20)
    hidden_dim : int — MLP hidden layer size (from config: 256)
    latent_dim : int — output dim (from config: 128)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ──────────
        x : Tensor of shape (batch_size, input_dim)

        Returns
        ──────
        Tensor of shape (batch_size, latent_dim)
        """
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Feature vector builder
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "norm_mass_earth",
    "norm_radius_earth",
    "norm_orbital_period_days",
    "norm_eq_temp_K",
    "norm_eccentricity",
    "norm_star_teff_K",
    "norm_star_luminosity",
    "norm_distance_pc",
    "is_hot",
    "is_icy",
    "is_giant",
    "has_short_year",
    "is_habitable",
]


def build_feature_vector(row: "pd.Series") -> np.ndarray:
    """
    Build a numeric feature vector from a planet DataFrame row.

    Uses normalised columns (prefixed `norm_`) plus boolean archetype flags.
    Missing values default to 0.5 (midpoint of normalised range) so the
    model receives a neutral signal rather than a zero that could be
    confused with a meaningful minimum.

    Returns
    ───────
    np.ndarray of shape (len(FEATURE_COLS),) with dtype float32
    """
    vec = []
    for col in FEATURE_COLS:
        val = row.get(col, 0.5)
        # Boolean → float
        if isinstance(val, bool) or (hasattr(val, "dtype") and np.issubdtype(type(val), np.bool_)):
            val = float(val)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.5
        if np.isnan(val):
            val = 0.5
        vec.append(val)
    return np.array(vec, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Dataset class for fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

class MythDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for planet→story pair fine-tuning.

    Each item is a dict with:
      input_ids      : LongTensor — tokenised input (prefix + story)
      attention_mask : LongTensor — 1 for real tokens, 0 for padding
      labels         : LongTensor — same as input_ids BUT prefix tokens
                       are set to -100 (masked from loss)
      feature_vec    : FloatTensor — normalised planet feature vector

    The label masking ensures the model is only trained to predict the
    *story* portion, not to reconstruct the feature prefix it is given
    as input.  This is the standard approach for conditional LM training
    (e.g. InstructGPT, FLAN).
    """

    def __init__(
        self,
        pairs: List[Dict],
        planet_df: "pd.DataFrame",
        tokeniser,
        max_length: int = 512,
    ):
        self.pairs     = pairs
        self.planet_df = planet_df
        self.tokeniser = tokeniser
        self.max_length = max_length

        # Build planet lookup: name → row series
        if planet_df is not None and "planet_name" in planet_df.columns:
            self.planet_lookup = {
                row["planet_name"]: row
                for _, row in planet_df.iterrows()
            }
        else:
            self.planet_lookup = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        planet_name  = pair["planet_name"]
        prefix_text  = pair["prefix"]                  # feature string
        story_text   = pair["story"]

        # Full input: "<myth> <archetype> <prefix>\n<story>"
        archetype    = pair.get("archetype", "myth")
        full_input   = f"<myth> <{archetype}> {prefix_text}\n{story_text}"

        # Tokenise prefix only (to know where story starts)
        prefix_only  = f"<myth> <{archetype}> {prefix_text}\n"
        prefix_enc   = self.tokeniser(
            prefix_only,
            add_special_tokens=False,
            return_tensors=None,
        )
        prefix_len   = len(prefix_enc["input_ids"])

        # Tokenise full sequence
        enc = self.tokeniser(
            full_input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)    # (seq_len,)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: copy of input_ids but mask prefix with -100
        labels = input_ids.clone()
        labels[:prefix_len] = -100                       # ignore prefix in loss
        # Also mask padding
        labels[attention_mask == 0] = -100

        # Feature vector
        row = self.planet_lookup.get(planet_name)
        if row is not None:
            feat_vec = torch.tensor(build_feature_vector(row), dtype=torch.float32)
        else:
            feat_vec = torch.zeros(len(FEATURE_COLS), dtype=torch.float32)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "feature_vec":    feat_vec,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ExoplanetMythosModel(nn.Module):
    """
    Complete Exoplanet Mythos Generator model.

    Wraps GPT-2 with:
      - Extended vocabulary (special tokens)
      - Optional MLP latent encoder for continuous feature conditioning
      - Forward pass that injects latent vector into first-position embeddings
      - Generation interface with style token prepending

    Inheritance: nn.Module (standard PyTorch base class).  Uses
    HuggingFace AutoModelForCausalLM internally.

    Forward pass
    ────────────
    1. Tokenise input text (done externally in Dataset).
    2. Compute MLP(feature_vec) → latent ∈ R^128.
    3. Retrieve token embeddings from GPT-2: emb ∈ R^(B, T, 768).
    4. Add latent[:, :128] to emb[:, 0, :128]  (only first token, first 128 dims).
       This is a minimal, non-destructive injection.
    5. Call GPT-2 forward with modified embeddings → logits.
    6. Compute cross-entropy loss on story tokens (labels != -100).
    """

    def __init__(self, cfg: dict, tokeniser):
        super().__init__()
        from transformers import AutoModelForCausalLM

        model_name = cfg["transformer"]["model_name"]
        latent_dim = cfg["transformer"]["latent_dim"]
        hidden_dim = cfg["transformer"]["latent_hidden"]
        n_features = len(FEATURE_COLS)

        log.info("Loading GPT-2 model: %s", model_name)
        self.gpt2 = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,   # needed for Lab A7 attention viz
        )
        self.gpt2.resize_token_embeddings(len(tokeniser))

        self.latent_encoder = PlanetLatentEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.latent_dim = latent_dim
        self.tokeniser  = tokeniser
        self.cfg        = cfg

        # Embedding dimension of GPT-2 (768 for gpt2, 1024 for gpt2-large)
        self.emb_dim = self.gpt2.config.n_embd
        assert latent_dim <= self.emb_dim, (
            f"latent_dim ({latent_dim}) must be ≤ GPT-2 embedding dim ({self.emb_dim})"
        )

        log.info("Model ready — GPT-2 params: %s  |  Latent encoder params: %d",
                 f"{sum(p.numel() for p in self.gpt2.parameters()):,}",
                 sum(p.numel() for p in self.latent_encoder.parameters()))

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
        feature_vec:    Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional latent feature conditioning.

        Parameters
        ──────────
        input_ids      : (B, T) — tokenised input sequences
        attention_mask : (B, T) — padding mask
        labels         : (B, T) — target token ids (-100 = masked from loss)
        feature_vec    : (B, F) — normalised planet feature vectors

        Returns
        ──────
        dict with keys: 'loss' (if labels provided), 'logits', 'attentions'
        """
        # 1. Get initial token embeddings from GPT-2 embedding layer
        inputs_embeds = self.gpt2.transformer.wte(input_ids)   # (B, T, E)

        # 2. Inject latent vector into position-0 embedding
        if feature_vec is not None:
            latent = self.latent_encoder(feature_vec)           # (B, latent_dim)
            # Additive injection — does not erase existing embedding signal
            inputs_embeds[:, 0, :self.latent_dim] += latent

        # 3. Forward through GPT-2 with modified embeddings
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss":       outputs.loss,
            "logits":     outputs.logits,
            "attentions": outputs.attentions,  # tuple of (B, H, T, T) per layer
        }

    @torch.no_grad()
    def generate_myth(
        self,
        prefix_text: str,
        feature_vec: Optional[np.ndarray] = None,
        style: str = "myth",
        max_new_tokens: int = 300,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.3,
        device: str = "cpu",
    ) -> str:
        """
        Generate a mythological story conditioned on a feature prefix string.

        Parameters
        ──────────
        prefix_text     : str — serialised planet features
        feature_vec     : np.ndarray — normalised numeric features for MLP
        style           : str — one of 'myth', 'fire', 'ice', 'gas', 'water'
        max_new_tokens  : int — maximum story tokens to generate
        temperature     : float — sampling temperature (higher = more random)
        top_p           : float — nucleus sampling probability mass cutoff
        top_k           : int — top-k sampling vocabulary restriction
        repetition_penalty : float — penalise repeated tokens

        Returns
        ──────
        str — generated myth story (decoded, without the prefix)

        Notes on Decoding Strategies
        ────────────────────────────
        We use nucleus (top-p) sampling rather than greedy decoding:
        - Greedy: always pick max-probability token → repetitive, degenerate
        - Top-k: sample from top-k tokens only → still can be repetitive
        - Nucleus (top-p=0.92): sample from the smallest set of tokens
          whose cumulative probability ≥ 0.92 → diverse, coherent output
        temperature rescales the logit distribution before sampling:
        - temp → 0: approaches greedy; temp → ∞: approaches uniform random
        - 0.85 is empirically good for creative text generation
        """
        self.eval()
        self.to(device)

        full_prompt = f"<myth> <{style}> {prefix_text}\n"
        enc = self.tokeniser(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        prefix_len     = input_ids.shape[1]

        # Inject latent vector into initial embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)
        if feature_vec is not None:
            fv_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0).to(device)
            latent = self.latent_encoder(fv_tensor)
            inputs_embeds[:, 0, :self.latent_dim] += latent

        outputs = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokeniser.pad_token_id,
            eos_token_id=self.tokeniser.eos_token_id,
        )

        # Decode only the newly generated tokens (strip the input prefix)
        generated_ids = outputs[0][prefix_len:]
        return self.tokeniser.decode(generated_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def get_attention_weights(
        self,
        input_text: str,
        device: str = "cpu",
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Return the attention weight tensors for a given input string.

        Used in the attention visualisation notebook (Lab A7).

        Returns
        ──────
        tokens    : List[str] — subword tokens
        attentions: List[Tensor] — one per layer, shape (n_heads, T, T)
        """
        self.eval()
        self.to(device)

        enc = self.tokeniser(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        input_ids = enc["input_ids"]
        outputs   = self.gpt2(
            input_ids=input_ids,
            output_attentions=True,
        )

        tokens      = self.tokeniser.convert_ids_to_tokens(input_ids[0].tolist())
        attentions  = [a.squeeze(0).cpu() for a in outputs.attentions]  # list of (H, T, T)
        return tokens, attentions


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, tokeniser) -> ExoplanetMythosModel:
    """Convenience factory function."""
    return ExoplanetMythosModel(cfg, tokeniser)


def load_checkpoint(checkpoint_path: str, cfg: dict, tokeniser) -> ExoplanetMythosModel:
    """
    Restore model from a saved checkpoint.

    The checkpoint contains: model state_dict, epoch, val_perplexity, config.
    """
    model = build_model(cfg, tokeniser)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded checkpoint from %s  (epoch %d, val_pp %.2f)",
             checkpoint_path, ckpt.get("epoch", "?"), ckpt.get("val_perplexity", float("inf")))
    return model
