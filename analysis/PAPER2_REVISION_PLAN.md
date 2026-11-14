# Paper 2: review response and controller redesign

Working audit, 10 October 2026. Sources: the pasted workshop decision and three reviews; arXiv 2609.00874v2; local manuscript; `HRM/models/subgoal_head.py`; `analysis/counterfactual_persistence_v2_within_pass/REPORT.md`. New experiments below are proposals, not results.

## What the reviews establish

The chairs explicitly cite unclear writing and presentation. Reviewers value the score/behavior/outcome distinction, but also identify substantive gaps. Better presentation alone would not resolve those gaps. The defensible empirical claim is that this particular controller did not beat the best tested forced schedule on its frozen checkpoints. This does not establish a general limit of adaptive revision.

| Concern | Required response | Priority |
|---|---|---|
| Final-split selection of best fixed schedule | Select the clock using calibration labels; freeze selection before final evaluation. Keep test-best clock as a separately labeled retrospective diagnostic. | First |
| Mean two emissions versus exactly two | Run exact-K=2 adaptive selection against exact-K=2 clocks; study variable counts separately with a shared cost/budget definition. | First |
| Local credit versus final outcome | Compare local-credit and endpoint-supervised controllers on the same frozen reasoner, observations, and action space. | First |
| Seed-0-only oracle and mechanisms | Enumerate all six clocks for every available seed; inspect the diverse seed explicitly. | First |
| Zero exact accuracy | Establish a task/model regime with nonzero complete-solution performance and meaningful timing headroom before expensive controller training. | Next study |
| Unspecified dataset/splits | Recover preprocessing provenance, base puzzle IDs, augmentation lineage, split construction and hashes. Check whether related augmentations cross splits. | First |
| Micro/macro inconsistencies | Produce one canonical per-seed, per-schedule table reporting both metrics and their separately selected winners; reconcile every figure against it. | First |
| Weak critic baseline/context | Report majority, pass-only, and within-pass comparisons, uncertainty and realized endpoint value. | First |
| Missing literature/workshop connection | Verify suggested citations and explain the specific methodological difference; add an agent-system experiment only if it tests the same decision-value question. | Writing / extension |

## Audit of the six observations

The implementation detaches all six observations and uses a zero-initialized affine sigmoid trigger. Detachment protects the reasoner from observation-path gradients; it does not make the observations sufficient for the decision.

| Observation | What it measures | Design concern and test |
|---|---|---|
| Cumulative cosine: worker displacement versus active goal | Directional adherence since emission | Adherence can increase toward a poor goal; cosine loses displacement magnitude. Compare against endpoint branch advantage. |
| Latest-pass cosine versus active goal | Direction of the latest update | Again goal-relative rather than task-relative; near-zero displacement needs inspection. Test magnitude and output-change information. |
| Change in cumulative cosine | Alignment trend | Related to the other geometry features; need not be redundant, but incremental value must be measured. Audit anchor/reset semantics. |
| Dwell divided by M | Age of the current commitment | Before the first optional replacement, dwell closely tracks pass. It can encode a useful clock while concealing weak state discrimination. Compare clock-only, state-only and clock-plus-state models. |
| Sigmoid(halt logit) minus sigmoid(continue logit) | Bounded halt/continue score difference | Its calibration and task relevance under fixed eight-pass execution must be established; it is not automatically revision confidence. |
| Active gate | Existing commitment's injection strength | Held constant between emissions; does not describe the quality of a proposed replacement. Audit within-position variation and predictive value. |

Main hypothesis: the observations mostly describe the current commitment, whereas the action requires comparing continuing it with replacing it. This is a plausible explanation, not yet demonstrated. An affine controller also cannot freely express interactions such as progress conditional on remaining compute. More capacity alone cannot fix an incorrect target or missing information.

## Controlled redesign

1. **Measure the opportunity first.** For every frozen checkpoint, evaluate all exact-K=2 clocks on calibration and final splits. Choose the deployable fixed comparator on calibration. Report final exact accuracy, macro/micro token accuracy, paired gaps, and diagnostic oracle headroom. Oracle labels are never policy inputs.
2. **Define a legal sequential policy.** One optional replacement after the mandatory pass-1 emission: once used, replacement is disabled; if unused, force it at pass 7. Include remaining budget and horizon explicitly. This matches the six-clock action space exactly.
3. **Construct endpoint supervision.** Branch PERSIST/REPLAN from identical cached states and continue to pass 8 under a specified shared continuation policy and remaining-budget constraint. Train on training episodes only. A branch advantage is conditional on its continuation policy; do not describe it as universal action value. Split by underlying puzzle where augmentation lineage permits.
4. **Separate target, observation, and capacity.** Compare local versus endpoint targets with the same six features; compare affine versus a small nonlinear controller; then add observations. Use clock-only and clock-plus-state controls throughout. Report endpoint improvement, not only classifier accuracy or AUROC.
5. **Test candidate-relative observations.** Candidate/active goal disagreement, displacement magnitude, output distribution change, predictive uncertainty and recent trajectory history are candidates. Validate their incremental within-position endpoint signal before selecting them. They remain proxies, not proven value estimates. Account for the compute needed to generate a candidate.
6. **Keep variable-budget allocation separate.** If variable counts are permitted, compare policies under the same explicit cost or expected-budget constraint and enumerate the corresponding legal schedule space where feasible. The six-clock oracle is not an upper bound for this larger space.

## Existing diagnostic worth preserving

The archived within-pass report contains 64 episodes and 384 states, with 114 held-out states. Its pass-plus-residual critic reports pass-2 AUROC 0.893 on only 19 held-out states and improves rare-positive classification over a majority baseline. This is limited evidence of conditional signal, not proof of useful endpoint control. All held-out positive states occur at pass 2. Check exactly which branch target the report uses before relating it to final-outcome value. The manuscript should distinguish the original adaptive trigger, global counterfactual critic, and residual critic.

## Writing structure

Lead with the concrete evaluation question and scoped outcome. Explain the persistent goal and eight-pass decision boundary with one diagram. Present the metric-aligned main table before mechanistic diagnostics. Then answer, in order: is there timing value; does the controller use state beyond time; does that use improve final outcomes? Put historical experiment chronology and secondary diagnostics in appendices. Avoid describing all results as replicated when only the headline comparison covers three seeds.

## Immediate next artifact

A canonical metric/schedule/provenance audit should precede manuscript edits or new training. Recover available episode matrices and checkpoint/data archives; identify which requested comparisons can be recomputed and which require new runs. Do not silently substitute new checkpoints for historical ones.
