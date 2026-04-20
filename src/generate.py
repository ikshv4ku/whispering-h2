"""
generate.py — Exoplanet Mythos Generator
=========================================
CST-106 Applications of NLP

PURPOSE
───────
End-to-end myth generation CLI.  Accepts either a planet name (looked up in
the processed dataset) or manual feature flags, generates a story, and
saves it.  This is the main demo script for the viva / presentation.

Usage Examples
──────────────
  # By planet name (looks up NASA data automatically)
  python src/generate.py --planet "55 Cnc e"

  # By name with style override
  python src/generate.py --planet "Kepler-442b" --style myth

  # Manual feature input (useful for hypothetical planets)
  python src/generate.py --temp 1800 --radius 12 --mass 350 --style fire

  # Generate multiple stories from test set
  python src/generate.py --batch 10 --style myth

  # Show all available style tokens
  python src/generate.py --list-styles

STYLE TOKENS
─────────────
  <myth>  — balanced mythological narrative (default)
  <fire>  — fire / heat / forge / Hephaestus / Surtur themes
  <ice>   — frost / winter / Skadi / Jormungandr / Niflheim themes
  <gas>   — vast / titan / Jörmungandr / oceanic / tempest themes
  <water> — sea / Poseidon / deep / abyssal themes

These tokens are used as conditional prefix tokens that steer generation
toward different mythological registers, demonstrating conditional LM
control (Lab A8 concept).

DECODING STRATEGY
──────────────────
We use top-p (nucleus) sampling with temperature scaling.  The rationale:
• Greedy decoding → repetitive and mode-collapsing (always picks the most
  probable token, converging to generic phrases quickly).
• Beam search → also repetitive for long creative text; optimises a joint
  probability that prefers short, safe sentences.
• Nucleus sampling → dynamically restricts the candidate set to the top-p
  probability mass, empirically producing the most creative, coherent text
  for language generation tasks (Holtzman et al. 2020).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def _load_config():
    from src.data_loader import load_config
    return load_config()


def _load_model(cfg, device="cpu"):
    """Load tokeniser and best model checkpoint."""
    from src.transformer_model import build_tokeniser, load_checkpoint
    tokeniser = build_tokeniser(cfg)
    ckpt_path = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"] / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}.\n"
            "Run `python src/train.py` first (or use --smoke_test for a quick demo)."
        )
    model = load_checkpoint(str(ckpt_path), cfg, tokeniser)
    return model, tokeniser


def _lookup_planet(planet_name: str, cfg) -> pd.Series:
    """Find a planet by name in the processed test/val/train datasets."""
    from src.data_loader import load_processed
    for split in ["test", "val", "train"]:
        try:
            df, _, _ = load_processed(split, cfg)
            if "planet_name" in df.columns:
                hit = df[df["planet_name"].str.lower() == planet_name.lower()]
                if len(hit) > 0:
                    return hit.iloc[0]
        except FileNotFoundError:
            continue
    return None


def _build_synthetic_row(args, cfg) -> pd.Series:
    """
    Build a synthetic planet row from CLI temperature / radius / mass arguments.
    Used when the user provides raw feature values instead of a planet name,
    demonstrating that the system is not limited to the NASA catalogue.
    """
    from src.data_loader import preprocess_planets
    import pandas as pd
    row = pd.Series({
        "planet_name":           "Custom Planet",
        "mass_earth":            args.mass,
        "radius_earth":          args.radius,
        "orbital_period_days":   args.period,
        "eq_temp_K":             args.temp,
        "eccentricity":          args.ecc,
        "star_spec_type":        args.star,
        "star_teff_K":           5500.0,
        "star_luminosity":       0.0,
        "distance_pc":           100.0,
        "discovery_method":      "Hypothetical",
    })
    feat_cfg = cfg["features"]
    row["is_hot"]        = row["eq_temp_K"]           > feat_cfg["temp_hot_threshold"]
    row["is_icy"]        = row["eq_temp_K"]           < feat_cfg["temp_cold_threshold"]
    row["is_giant"]      = row["radius_earth"]        > feat_cfg["radius_giant_threshold"]
    row["has_short_year"]= row["orbital_period_days"] < feat_cfg["period_short_threshold"]
    row["is_habitable"]  = (feat_cfg["habitable_temp_min"] <= row["eq_temp_K"] <= feat_cfg["habitable_temp_max"]
                            and row["radius_earth"] <= feat_cfg["habitable_radius_max"])
    # Simple normalisation with typical reference ranges
    row["norm_mass_earth"]            = min(row["mass_earth"]            / 500, 1.0)
    row["norm_radius_earth"]          = min(row["radius_earth"]          / 25,  1.0)
    row["norm_orbital_period_days"]   = min(row["orbital_period_days"]   / 365, 1.0)
    row["norm_eq_temp_K"]             = min(row["eq_temp_K"]             / 5000, 1.0)
    row["norm_eccentricity"]          = row["eccentricity"]
    row["norm_star_teff_K"]           = 0.5
    row["norm_star_luminosity"]       = 0.5
    row["norm_distance_pc"]           = 0.2
    row["spec_class_letter"]          = row["star_spec_type"][0] if row["star_spec_type"] else "G"
    return row


def generate_story(
    planet_name: str = None,
    planet_row: pd.Series = None,
    style: str = "myth",
    cfg: dict = None,
    model=None,
    device: str = "cpu",
    verbose: bool = True,
) -> str:
    """
    Core generation function.  Either planet_name (looked up) or planet_row
    (pre-built Series) must be provided.

    Returns the generated story string.
    """
    from src.data_loader import serialise_planet_features, build_feature_vector

    if cfg is None:
        cfg = _load_config()
    if model is None:
        model, _ = _load_model(cfg, device)

    if planet_row is None and planet_name:
        planet_row = _lookup_planet(planet_name, cfg)
        if planet_row is None:
            log.warning("Planet '%s' not found in dataset. Using default features.", planet_name)
            planet_row = _build_synthetic_row(
                argparse.Namespace(
                    mass=50, radius=3, period=20, temp=700, ecc=0.05, star="G"
                ),
                cfg
            )
            planet_row["planet_name"] = planet_name

    prefix   = serialise_planet_features(planet_row)

    # Override style from archetype if not specified
    archetype = planet_row.get("myth_archetype", style)
    if style not in ["myth", "fire", "ice", "gas", "water"]:
        style = archetype

    fvec = None
    try:
        from src.transformer_model import build_feature_vector as _bfv
        fvec = _bfv(planet_row)
    except Exception:
        pass

    gen_cfg = cfg["generation"]
    story = model.generate_myth(
        prefix_text=prefix,
        feature_vec=fvec,
        style=style,
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        top_k=gen_cfg["top_k"],
        repetition_penalty=gen_cfg["repetition_penalty"],
        device=device,
    )

    if verbose:
        pname = planet_row.get("planet_name", "Unknown")
        print("\n" + "═" * 60)
        print(f"  PLANET : {pname}")
        print(f"  STYLE  : <{style}>")
        print(f"  ARCHETYPE: {archetype}")
        print(f"  TEMP   : {planet_row.get('eq_temp_K', '?')} K")
        print(f"  RADIUS : {planet_row.get('radius_earth', '?')} R_earth")
        print(f"  IS HOT : {planet_row.get('is_hot', '?')} | IS ICY: {planet_row.get('is_icy', '?')}")
        print("─" * 60)
        print(f"\n{story}\n")
        print("═" * 60 + "\n")

    # Save story
    stories_dir = PROJECT_ROOT / cfg["paths"]["stories_dir"]
    stories_dir.mkdir(parents=True, exist_ok=True)
    pname_safe = str(planet_row.get("planet_name", "unknown")).replace(" ", "_").replace("/", "-")
    out_path = stories_dir / f"{pname_safe}_{style}.txt"
    out_path.write_text(
        f"Planet: {planet_row.get('planet_name', 'Unknown')}\n"
        f"Style: <{style}>\n"
        f"Prefix: {prefix}\n\n"
        f"{story}",
        encoding="utf-8",
    )
    log.info("Story saved to %s", out_path)
    return story


def batch_generate(n: int, style: str, cfg: dict, model, device: str) -> None:
    """Generate stories for the first n test planets and save all."""
    from src.data_loader import load_processed
    test_df, test_pairs, _ = load_processed("test", cfg)
    planet_lookup = {}
    if "planet_name" in test_df.columns:
        planet_lookup = {r["planet_name"]: r for _, r in test_df.iterrows()}

    count = 0
    for pair in test_pairs[:n]:
        pname = pair["planet_name"]
        row = planet_lookup.get(pname)
        if row is None:
            continue
        generate_story(planet_row=row, style=style, cfg=cfg, model=model,
                       device=device, verbose=True)
        count += 1
    log.info("Generated %d stories.", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mythological stories from exoplanet data."
    )
    # Planet selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--planet", type=str, default=None,
                       help="Exact planet name to look up (e.g. '55 Cnc e')")
    group.add_argument("--batch", type=int, default=None,
                       help="Generate stories for N test planets")

    # Manual feature override
    parser.add_argument("--temp",   type=float, default=700,  help="Equilibrium temperature (K)")
    parser.add_argument("--radius", type=float, default=3.0,  help="Planet radius (Earth radii)")
    parser.add_argument("--mass",   type=float, default=50.0, help="Planet mass (Earth masses)")
    parser.add_argument("--period", type=float, default=20.0, help="Orbital period (days)")
    parser.add_argument("--ecc",    type=float, default=0.05, help="Orbital eccentricity")
    parser.add_argument("--star",   type=str,   default="G",  help="Host star spectral type")

    # Generation
    parser.add_argument("--style",  type=str, default="myth",
                        choices=["myth", "fire", "ice", "gas", "water"],
                        help="Mythological style token")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device: cpu | cuda | mps")
    parser.add_argument("--list-styles", action="store_true",
                        help="Print available style descriptions and exit")
    args = parser.parse_args()

    if args.list_styles:
        print("\nAvailable style tokens:")
        print("  <myth>  — balanced mythological narrative (default)")
        print("  <fire>  — fire / heat / forge / Hephaestus / Surtur themes")
        print("  <ice>   — frost / winter / Skadi / Jormungandr / Niflheim themes")
        print("  <gas>   — vast / titan / oceanic / tempest themes")
        print("  <water> — sea / Poseidon / deep / abyssal themes")
        sys.exit(0)

    cfg = _load_config()
    try:
        model, _ = _load_model(cfg, args.device)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("To run a demo without training, use the n-gram baseline:")
        print("  python src/baseline_models.py --ngram-only")
        sys.exit(1)

    if args.batch:
        batch_generate(args.batch, args.style, cfg, model, args.device)
    elif args.planet:
        generate_story(planet_name=args.planet, style=args.style,
                       cfg=cfg, model=model, device=args.device)
    else:
        # Manual features mode
        planet_row = _build_synthetic_row(args, cfg)
        generate_story(planet_row=planet_row, style=args.style,
                       cfg=cfg, model=model, device=args.device)
