# pmcma — Private-Markets CMA Methodology Demonstrator

A config-driven pipeline that builds capital market assumptions (CMAs) for an
illiquid/private asset class and places them on a **consistent, comparable basis**
with a public asset class — the core problem in whole-portfolio strategic asset
allocation across public and private markets.

> **This is a methodology demonstrator, not a proprietary CMA.**
> It runs on public proxies and synthetic series, not gated vendor data
> (Burgiss/Preqin/Cambridge). Every output is labeled accordingly. The point is to
> demonstrate that the *methods* are correct and reproducible on transparent inputs —
> which is exactly what a research process must establish before vendor data is applied.

---

## Why these four modules

A naive private-markets CMA — taking reported appraisal returns at face value — is
wrong in four specific, well-understood ways. Each module addresses one:

| Problem | Symptom if ignored | Module |
|---|---|---|
| **Appraisal smoothing** | Private risk looks artificially low; near-zero correlation to public markets | `desmooth` |
| **Lumpy, GP-controlled cash flows** | "Invest $X today" is fiction; you commit and get called/distributed over years | `pacing` |
| **Illiquidity under stress** | Denominator effect; calls continue while distributions dry up in a drawdown | `liquidity` |
| **Manager dispersion** | "The asset-class return" is meaningless without a selection assumption | `dispersion` |

The synthesis step assembles these into one table where a private asset class sits
next to a public one on the same basis — de-smoothed risk, pacing-aware exposure,
liquidity-stressed sustainable pace, and a dispersion-aware return band.

---

## Modules

- **`desmooth.py`** — Recovers true volatility and public-market correlation from
  serially-correlated appraisal returns. Geltner (1991) first-order and
  Getmansky–Lo–Makarov (2004) MA unsmoothing. *The headline result:* reported vs.
  de-smoothed risk, side by side.
- **`pacing.py`** — Takahashi–Alexander deterministic cash-flow model. Projects
  capital calls, distributions, and NAV for a fund and aggregates staggered vintages
  to a steady-state invested percentage. Answers: *to hold X% invested, commit how
  much and when?*
- **`liquidity.py`** — Couples the pacing engine to a public-market shock to model the
  denominator effect and liquidity coverage, and solves for the **maximum sustainable
  commitment pace** that survives the stress.
- **`dispersion.py`** — Simulates the return distribution an allocator actually faces
  given manager selection, converting quartile inputs into an expected-return band
  rather than a single point.
- **`cma.py`** — Assembles the comparable public-vs-private CMA table.
- **`narrative.py`** — Human-in-the-loop generation of a short research note over the
  finished table, plus governed extraction of cash-flow inputs from GP report PDFs.
  Both steps pause for review and refuse to auto-accept: with no interactive terminal
  they raise rather than proceed.
- **`pipeline.py`** — Orchestrator. `run(config)` executes the full sequence and writes
  the labeled table and note. New inputs in → full research output out, reproducibly.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tests; add ,full for narrative/plotting later
pytest                            # includes the de-smoothing round-trip test
python -m pmcma.pipeline          # runs config/default.yaml → outputs/
```

Configuration is entirely in `config/default.yaml` — asset series, de-smoothing method,
pacing parameters, stress scenario, and dispersion inputs. No parameters are hard-coded
in module logic.

---

## Status

**Working:** all four analytical modules built and unit-tested; config-driven pipeline
runs end-to-end and produces a labeled CMA table carrying de-smoothed risk, pacing,
`max_sustainable_pace`, and the manager-dispersion band. `narrative.py` is built — the
governed note and GP-report steps run behind their human-review pause. Full suite green.

`narrative.enabled` ships **false** in `config/default.yaml`: an enabled run makes a live
API call gated on that review pause, so it needs `ANTHROPIC_API_KEY` and a TTY. Set it
true for an interactive demo run.

**Pending (not build-order-critical):**
1. Expanded real-proxy datasets (currently synthetic + limited public proxies).

---

## Known limitations & bugs surfaced by running

Passing unit tests is not correctness. The following issues surfaced only by driving
the full pipeline on real inputs — documented here because *how* they surfaced is the
point:

- **Geometric vs. arithmetic return on the de-smoothed series** *(methodology).*
  De-smoothing changes the return path; the annualization/compounding convention has to
  be applied consistently to the recovered series, not carried over from the reported one.
  A silent methodology error, invisible to unit tests checking mechanics.
- **`pyyaml` mis-tiered as an optional dependency.** The pipeline can't load config
  without it; it belongs in core, not the `full` extra. Surfaced on a clean install.
- **Silent Series-name row collision** in table assembly — identically-named Series
  merged/overwrote rather than erroring. Correct-looking output, wrong underlying rows.
- **Bisection bracket returned a breaching pace as "safe."** The `max_sustainable_pace`
  solver's bracket edge could report a commitment pace that in fact breaches the liquidity
  constraint — a wrong answer in the one direction that matters for an allocator. Fixed;
  verify the scalar against a hand-checked case before relying on it.

These are tracked, not hidden. The de-smoothing methodology fix and the bisection fix are
the two that affect headline numbers and were prioritized accordingly.

---

## Scope note

This artifact is a credibility demonstrator built to show that private-markets CMA
methodology can be implemented as a repeatable, governed, reproducible workflow. It is
not investment advice and its outputs are illustrative, derived from public/synthetic
proxies.
