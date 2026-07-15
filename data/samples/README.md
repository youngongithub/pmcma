# Sample data — SYNTHETIC

These series are **synthetic, transparent demo inputs — not vendor data**
(no Burgiss/Preqin/Cambridge). They exist so the methodology runs end-to-end on
reproducible inputs.

- `public_quarterly.csv` — synthetic public-equity quarterly returns (`date,ret`).
- `private_proxy_quarterly.csv` — synthetic private-equity proxy, stored in its
  **reported (appraisal-smoothed)** form; Module 1 de-smooths it.

Regenerate deterministically with: `python scripts/make_samples.py`
