# Replication Archive: Adaptive SIMEX for Poisson Regression

This directory contains the complete, reproducible pipeline for the
simulation studies (Section 5) and the real-data application
(Section 6) of

> G. Garg and R. Mishra, "Adaptive SIMEX for Poisson Regression with
> Classical Measurement Error: Theory, Methods, and Applications."

**Master seed: 20260905.** Every Monte Carlo repetition and the NHANES
analysis derive their randomness from
`numpy.random.SeedSequence(20260905, spawn_key=...)`, so all tables and
figures are bit-reproducible on any machine with numpy >= 1.25.

## Contents

```
code/
  asimex_lib.py           method library: batched Poisson IRLS, SIMEX,
                          Adaptive SIMEX (CV + 1-SE rule + argmin variant),
                          ridge second stage (Sec. 3.2), Nakamura corrected
                          score, regression calibration, analytic sandwich
                          variance (data + O(1/B) simulation components)
  run_sims.py             Monte Carlo driver (all scenarios, multiprocessing)
  make_tables_figures.py  aggregates raw results -> Tables 1-5, Figures 1-3
  nhanes_analysis.py      NHANES 2017-18 application -> nhanes_table.tex,
                          fig4_sensitivity.pdf/png, nhanes_results.json
  nhanes_transfer.py      NHANES 2015-16 transfer application ->
                          transfer_table.tex, fig8_transfer.*
  validate_sandwich.py    analytic-vs-bootstrap sandwich validation
data/
  *.XPT                   NHANES 2017-2018 files (CDC, public domain):
                          DEMO_J, BMX_J, WHQ_J, RXQ_RX_J
results/
  raw/*.npz               per-chunk Monte Carlo output (one file per
                          scenario/cell/100 reps)
  table1.tex ... table5.tex   manuscript tables (booktabs)
  table1_small.tex        appendix table (n = 100, 300)
  fig1_bias.pdf/png       Figure 1: bias vs sigma^2 by n
  fig2_coverage.pdf/png   Figure 2: coverage heatmap (Laplace misspec.)
  fig3_time.pdf/png       Figure 3: computation time vs p
  fig4_sensitivity.pdf/png   Figure 6: NHANES sigma^2 sensitivity
  fig5_simex_curve.pdf/png   Figure 2: SIMEX curve diagnostic
  fig6_rmse.pdf/png          Figure 3: RMSE vs sigma^2 panels
  fig7_nhanes_forest.pdf/png Figure 7: NHANES coefficient forest plot
  fig8_transfer.pdf/png      Figure 8: transfer application estimates
  transfer_table.tex         Table 8: NHANES 2015-16 transfer results
  nhanes_results.json, transfer_results.json   estimates for figures
  nhanes_table.tex        application table
  section5_simulations.tex    REPLACEMENT Section 5 (real numbers)
  section6_application.tex    REPLACEMENT Section 6 (NHANES application)
  revisions_manuscript_text.tex  REPLACEMENT Abstract, Sec 1.2-1.3, 3.3,
                              7.4-7.5, 8-9 + ten-item change log mapping
                              every superseded draft claim to evidence
  revisions_preview.pdf       the revisions compiled for reading
  verification_master.tex/pdf compiles Sections 5-6 + tables + figures
adaptsimex/                 R PACKAGE (CRAN-ready artifact)
  DESCRIPTION, NAMESPACE    Authors@R, GPL-3, stats/graphics only
  R/                        adaptive_simex(), cv_select(), simex_curve(),
                            simex_sandwich(), nakamura_corrected_score(),
                            regression_calibration()
  man/                      6 Rd documentation pages
  vignettes/adaptsimex.Rnw  Sweave quick-tour vignette (builds)
  tests/                    base-R runner + testthat suite (all pass)
  adaptsimex_0.1.0.tar.gz   source tarball; R CMD check: 0 E / 0 W /
                            1 NOTE (local tidy/V8 tooling only; examples
                            runnable at 0.6 s, vignette builds in 60 s)
manuscript/
  poisson_paper_revised.tex/pdf   COMPLETE REVISED MANUSCRIPT (42 pp.,
                                  elsarticle 3p, JoE format):
                                  revised abstract/Intro, Secs 2-3, Sec 4
                                  theory (Thms 1-4, Lems 1-6), real Secs
                                  5-6, revised Secs 7-9, Appendix A proofs,
                                  APA references. Compiles clean (pdflatex x2)
  part1.tex, part2.tex, part3.tex   source parts (Secs 1-3, 4, 7-9)
  section5.tex, section6.tex        section copies with resolved paths
  appendix_proofs.tex               Appendix A (generated from proofs doc)
asimex_proofs.tex/pdf    standalone corrected proofs (source of Appendix A)
poisson_paper.pdf        original June 2026 draft (superseded)
```

## Reproducing everything

```bash
cd code
python3 run_sims.py all --chunk 100 --workers 16     # ~22 min on M-series Mac
python3 make_tables_figures.py                       # Tables 1-5, Figs 1-3
python3 validate_sandwich.py                         # Sec. 5.7 validation
python3 nhanes_analysis.py                           # Sec. 6 (needs pandas)
```

Scenario IDs: `S1` (4 x 3 cells), `S2` (Laplace misspecification),
`S3` (timing, p in {5,15,30,50}), `T4` (negative binomial), `T5`
(estimated sigma^2 from m = 2 replicates).

Applications: `nhanes_analysis.py` (2017-18 cycle) and
`nhanes_transfer.py` (2015-16 cycle, out-of-sample transfer of the
calibrated error variance). The RXQ prescription files hold one row
per medication record, so both scripts deduplicate to person level
(`drop_duplicates("SEQN")`) and assign RXDCOUNT = 0 to persons with no
medication record before the complete-case filters (final samples
n = 5,628 and n = 6,100).

## Method configuration

| Parameter | Value |
|---|---|
| SIMEX replicates B | 100 |
| CV folds | 5 |
| CV replicates per fold-curve B_cv | 25 |
| lambda_max grid | {0.5, 1.0, 1.5, 2.0, 2.5} |
| degree grid | {1, 2, 3} |
| Standard SIMEX grid | {0, 0.5, 1, 1.5, 2}, quadratic |
| Ridge second stage | gamma = 0.01, degree + 2 |
| SEs | analytic sandwich (Theorem 2 / Cor. 1 of the proofs) |

## Key empirical findings (see Section 5 for detail)

1. At small error (sigma^2 = 0.1), A-SIMEX matches the best competitor
   (bias -0.011 vs naive -0.066 at n = 1000) with valid coverage
   (0.84-0.91).
2. At large error (sigma^2 = 1.0), residual polynomial-extrapolation
   bias dominates for ALL SIMEX variants (std -0.149, A-SIMEX 1-SE
   -0.191); the observed-scale CV criterion systematically
   under-extrapolates because its population minimizer is the naive
   pseudo-parameter beta(1), not beta_0 (see Lemma 1 of the proofs).
   This replaces the draft's unverified claim that A-SIMEX attains
   near-oracle bias at all error levels.
3. The efficiency premium of extrapolation is large and quantifiable
   (RMSE ratio to oracle ~ 2.6 at sigma^2 = 0.1, n = 1000), consistent
   with the weight norm sum |c_j| ~ 7.8 of the quadratic extrapolant
   (Theorem 3).
4. The analytic sandwich recovers 66-91% of the bootstrap variance
   (first-order under-statement, improving in n); software provides a
   bootstrap option for small samples.
5. Corrections are robust to (a) Gaussian-assumption misspecification
   (Laplace DGP), (b) negative-binomial overdispersion with robust SEs,
   and (c) replacing known sigma^2 by the replicate-based estimate.
6. NHANES application: all corrections agree (+0.50-0.51 per 10 kg,
   s.e. 0.026 vs naive +0.490); the measured-weight anchor (+0.468)
   sits 1.5 SE below, consistent with nonclassical (truth-correlated)
   self-report error; estimates are stable across a fourfold
   sigma^2 sensitivity range.
