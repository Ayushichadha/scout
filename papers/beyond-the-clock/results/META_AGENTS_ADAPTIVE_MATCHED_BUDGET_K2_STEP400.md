# Adaptive Matched-Budget K=2 Audit — Step 400

## A. Checkpoint provenance

- Adaptive checkpoint: `/Users/ayushi/Documents/hrm_research/experiments/meta_agents_adaptive_400step/20260820T101100Z_adaptive_eta0_seed0_400step/checkpoints/step_400`
- SHA-256: `165586ca87a5ca473b342ccb4342afd72890817498c87319964356aa37313889`
- Training: 400 optimizer steps, seed 0, eta=0, M=8, CPU.
- Frozen reconstructed baselines: P=4 `2e5d4ba89ed23fc70232d5c0603b9c5ab6cd2a5c6fecd410a83d19268d614f49`, P=6 `dace3b0dbb1ea69fcb8a1e0ac707c78267a24950f52364ef7f882c3c95aeb1e7`.
- Calibration/final split: 921 / 3,686 episodes; identical to the prior diagnostic; zero overlap.
- Every checkpoint was state-hashed before/after evaluation and copied into `frozen_inputs/`.

## B. 400-step beta/ranking audit

At the natural deterministic theta=0.5 reference trajectory, beta has count 27642, mean 0.498048024, std 0.000100464, range [0.497820377, 0.498337328], and median 0.498022765.

Position explains 96.760% of total beta variance; 3.240% remains as residual same-position score variation. This residual is not labeled “state intelligence.”

Time-only versus full-logit correlation is Pearson 0.9420 and Spearman 0.9776. Progress-only std/range are 1.650e-04/9.698e-04, versus dwell-only 4.743e-04/1.389e-03. Detailed per-feature scales, ranges, and associations are in `feature_beta_statistics.json`; fixed-pass action groups are in `clock_likeness.json`.

## C. Calibration

- Selection objective: intervention budget only, target total K=2.
- Selected theta: `0.498200`.
- Calibration mean K: 2.000000.
- Task accuracy/loss were unavailable to the selector.

| Threshold | Mean K | Absolute error |
|---:|---:|---:|
| 0.498199 | 2.001086 | 0.001086 |
| 0.498200 | 2.000000 | 0.000000 |
| 0.498201 | 1.995657 | 0.004343 |

## D. Final adaptive behavior

- Final episodes: 3,686
- Mean total K / adaptive interventions: 2.000814 / 1.000814
- Hard intervention rate: 0.166802
- Token/exact accuracy: 0.467015 / 0.000000
- LM loss per executed pass: 1.957886
- Mean/median dwell: 3.498576 / 1.0
- Unique schedules / entropy: 4 / 0.077601 bits
- Dominant schedule: [1, 2] (99.19%)
- Replacement old/new-goal cosine: {'count': 3689, 'mean': 0.8079508662902427}

## E. Matched 400-step comparison

| Condition | Steps | Mean K | Token accuracy | Exact accuracy | LM loss/pass | Dominant schedule |
|---|---:|---:|---:|---:|---:|---|
| fixed_p4 | 400 | 2.000000 | 0.481321 | 0.000000 | 1.958075 | [1, 4] |
| fixed_p6 | 400 | 2.000000 | 0.485595 | 0.000000 | 1.957461 | [1, 6] |
| adaptive_calibrated | 400 | 2.000814 | 0.467015 | 0.000000 | 1.957886 | [1, 2] |

Descriptive differences:

- Adaptive - P4 token accuracy: -0.014307
- Adaptive - P6 token accuracy: -0.018581
- P6 - P4 token accuracy: +0.004274
- Adaptive - P4 LM loss/pass: -0.000189
- Adaptive - P6 LM loss/pass: +0.000425

## F. Scientific interpretation

**Outcome C: after equal 400-step training, this checkpoint does not convert its replanning ranking into improved matched-budget performance.**

1. **Calibration quality:** calibration K is 2.000000 and final K is 2.000814; budget transfer is successful.
2. **Ranking/state dependence:** position explains 96.760% of beta variance and the time-only/full Spearman correlation is 0.9776. The score is predominantly temporal. The 3.240% residual and mixed pass-2 actions show descriptive same-position variation, not causal feature intelligence.
3. **Realized schedule adaptivity:** 99.19% of final episodes use `[1,2]`; entropy is only 0.077601 bits. State-dependent exceptions exist but are behaviorally rare.
4. **Task performance:** adaptive token accuracy is 0.0143 below P=4 and 0.0186 below P=6. LM loss/pass is essentially matched (adaptive-P4 -0.000189; adaptive-P6 +0.000425), so the former 96-vs-400 training-budget confound no longer explains the accuracy deficit.
5. **Limitations:** theta was calibrated externally, this is one seed-0 checkpoint comparison, and the P=4/P=6 weights are deterministic reconstructions of runs whose original weights were not saved. No significance claim or causal feature attribution is made.

## G. Invariants and archival status

- All conditions used exactly eight executed refinement passes per completed episode.
- Adaptive eligibility was exactly six decisions per episode; pass 1 was forced and no pass-8 emission occurred.
- Every adaptive episode's dwell lengths summed to seven, and its dwell-segment count equaled its total intervention count.
- Calibration/final overlap was zero and the ordered split digests match the prior diagnostic exactly.
- Original checkpoint hashes and loaded model-state hashes matched before and after evaluation.
- All serialized numeric artifacts passed finite-value checks; no compute or terminal-emission violation occurred.
- The full repository test suite passed: 50 tests.
- Verified copies of all three checkpoints, configs, source snapshots, run summaries, and training logs are stored under `frozen_inputs/` with `MANIFEST.json` hashes.
