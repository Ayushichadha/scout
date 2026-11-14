# Proposed software-agent transfer study

The primary experiment tests goal persistence and revision in HRM. This protocol describes a possible transfer pilot.

## Research question

Does observed execution state improve persist-versus-replan decisions at a
matched remaining budget? The primary endpoint is issue resolution under the
benchmark grader. Local scores are secondary diagnostics.

## Ordered milestones

1. Audit overlap with Executable Prefix Values (arXiv:2610.17633),
   Fail-fast, Restart-smart (arXiv:2608.03222), and Continue or Replan?
   (arXiv:2608.03483). Verify intervention definitions and released code.
2. Pin mini-SWE-agent and its model configuration. Reproduce one SWE-bench
   Verified issue in an isolated environment with executable grading.
3. Snapshot a pre-decision state and restore two independent branches.
   Verify conversation, tracked and untracked files, working directory,
   environment, and remaining budget. Record snapshot hashes.
4. Pilot 20–30 development issues. Repeat a subset of branch continuations
   to assess sampling variation. Choose main-study size from observed
   disagreement and measured cost.
5. Fit time-only and time-plus-state supervisors using identical paired
   outcomes. Select fixed and reactive baselines on validation issues.
6. Freeze policies and evaluate complete runs on untouched issues.

## Intervention and accounting

Persist permits the worker's next ordinary action. Replan requests a revised
plan using the existing execution evidence and preserves the working tree.
The replan call consumes budget. Initially allow one intervention per run.
Record model calls, input and output tokens, wall time, and monetary cost when
available. Pin model identifier, sampling settings, agent commit, task version,
image digest, prompts, and budget in every run manifest.

Split by issue before collecting supervisor fitting data or tuning. Keep
benchmark grading tests unavailable to policy features. Record every sampled
issue and every failed infrastructure run, including retry disposition.

Report only-replan, only-persist, both-solve, and neither-solve counts with
distinct-issue denominators. Evaluate paired resolution differences and
confidence intervals by issue. Report rescues and losses against time-only,
fixed-schedule, always-persist, and reactive policies.

## Implementation record

2026-11-13 — Protocol established. Local preflight found no Docker executable.
Official SWE-bench evaluation uses Docker, with hosted evaluation also
documented. Environment provisioning and model access must be established
before claiming a graded software-agent experiment. No software-agent issue
has been executed or graded in this study yet.

Code and concise milestone records belong in Git. Raw trajectories, images,
checkpoints, and detailed result tables remain outside committed content.

## References

- https://mini-swe-agent.com/latest/
- https://www.swebench.com/SWE-bench/guides/evaluation/
- https://arxiv.org/abs/2610.17633
- https://arxiv.org/abs/2608.03222
- https://arxiv.org/abs/2608.03483
