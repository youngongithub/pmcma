"""config in → CMA table + narrative out. The 'repeatable, governed workflow'."""
from __future__ import annotations
import yaml
from pathlib import Path
from . import io, desmooth, pacing, cma
# NOTE: liquidity (M3), dispersion (M4) and narrative are imported *lazily*
# inside their guarded branches below. They may not exist yet, so a top-level
# import would break the MVP pipeline on `import`. They light up automatically
# once the module file exists and its config section sets `enabled: true`.


def _write_labeled_csv(df, path: str, title: str, notes: str = "") -> None:
    """Write a CSV whose header carries the source label + 'methodology
    demonstrator' (design principle #5). Header lines are '#'-commented so the
    file still round-trips via pandas.read_csv(comment='#')."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(f"# {title}\n")
        if notes:
            f.write(f"# {notes}\n")
        df.to_csv(f)


def run(config_path: str = "config/default.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    label = cfg["data_source_label"]
    run_name = cfg["run_name"]
    ppy = cfg["pacing"]["periods_per_year"]

    pub = io.load_returns(cfg["public_asset"]["return_series"])
    priv = io.load_returns(cfg["private_asset"]["return_series"])
    # label series from config so the CMA table rows are the asset names, not "ret"
    pub.name = cfg["public_asset"]["name"]
    priv.name = cfg["private_asset"]["name"]  # priv_ds inherits this name below

    priv_ds = desmooth.geltner_unsmooth(priv) if cfg["desmooth"]["method"] == "geltner" \
        else desmooth.glm_unsmooth(priv, cfg["desmooth"]["glm_lags"])
    risk_tbl = desmooth.summarize_risk(priv, priv_ds, pub, ppy)

    pparams = pacing.PacingParams(**{k: cfg["pacing"][k] for k in (
        "commitment", "fund_life", "bow_factor", "growth_rate",
        "contribution_rate", "yield_rate", "periods_per_year")})
    program = pacing.project_program(cfg["pacing"]["program_commitments"]["schedule"], pparams)

    liq_tbl = None
    liq_cfg = cfg.get("liquidity") or {}
    if liq_cfg.get("enabled", False):
        from . import liquidity  # M3 — lazy
        stressed = liquidity.apply_shock(program, **liq_cfg["shock"])
        liq_tbl = liquidity.liquidity_coverage(stressed, liq_cfg["liquid_sleeve_weight"])

    disp = None
    if cfg.get("dispersion", {}).get("enabled", False):
        from . import dispersion  # M4 — lazy
        d = cfg["dispersion"]
        mu, sigma = dispersion.fit_lognormal_from_quartiles(
            d["top_quartile"], d["median"], d["bottom_quartile"])
        sims = dispersion.simulate_selection(mu, sigma, d["n_sims"])
        disp = dispersion.dispersion_band(sims, horizon_years=10)

    table = cma.assemble_cma_table(pub, priv_ds, program, liq_tbl, disp, ppy, label)

    Path("outputs").mkdir(exist_ok=True)
    _write_labeled_csv(table, f"outputs/{run_name}_cma.csv",
                       table.attrs["title"], table.attrs["notes"])
    _write_labeled_csv(risk_tbl, f"outputs/{run_name}_desmooth.csv",
                       f"{label} — de-smoothing risk recovery — methodology demonstrator")

    # Narrative: human-in-the-loop (principle #3). Stays off until narrative.py
    # exists AND implements the review PAUSE — never auto-accept generated text.
    nar_cfg = cfg.get("narrative", {})
    if nar_cfg.get("enabled", False):
        from . import narrative  # lazy
        note = narrative.generate_narrative(table, nar_cfg["model"],
                                            nar_cfg.get("require_review", True))
        Path(f"outputs/{run_name}_note.md").write_text(note)

    print(f"[pmcma] {run_name}: wrote outputs/{run_name}_cma.csv (+ _desmooth.csv). "
          f"Label: {label}")


if __name__ == "__main__":
    run()
