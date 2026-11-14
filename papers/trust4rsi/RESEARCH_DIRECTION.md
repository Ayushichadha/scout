# Adaptive persistence and revision

The question is whether current state improves subgoal revision decisions beyond
elapsed time at matched remaining computation. HRM with FeUdal goals is the
mechanism foundation. Stateful software tasks are the transfer direction selected
by the author. TRM provides a possible architecture replication.

## Existing software experiments

- run_tool_revision_pilot.py tests mandatory service switching.
- run_optional_tool_revision.py permits zero or one switch. Five runs averaged
  83.32% completion for state-aware control and 83.30% for the reactive rule.
- run_sqlite_revision.py executes actual SQLite transfers and cursor restoration,
  with synthetic service faults. Three fresh comparisons against reactive control
  had gaps of 0.45–0.75 percentage points, each interval including zero.
- run_feudal_sqlite_bridge.py conditions a learned worker on cached Scout goals
  and permits a goal refresh. Two seeds scored 81.2% and 81.0% for state-based
  refresh, versus 81.0% and 81.2% for fixed refresh. The worker uses endpoint
  supervision, without the directional auxiliary loss.

These tasks provide implemented interventions, endpoint scoring, and reproducible
traces. The goal-refresh results do not yet demonstrate a consistent benefit.
The later coding-agent protocol is a candidate extension requiring its own
feasibility assessment.

## Current execution order

1. Complete the bounded HRM competence run and inspect its saved predictions.
2. Establish broader development competence with source-separated data.
3. Collect endpoint persist/refresh pairs for the frozen worker.
4. Train matched time-only and state-aware supervisors on identical paired data.
5. Evaluate whole trajectories against fixed and reactive rules, with paired
   uncertainty by problem.
6. Apply the same comparison to the existing SQLite goal-refresh task, first
   measuring whether refresh offers enough distinct successful outcomes.

Software-task work remains part of this plan throughout HRM development.
Scaling the coding-agent pilot depends on its scientific contribution and
available execution resources.

## Reviewer knowledge in the experiment

Russell and Wefald (1991) motivate downstream decision utility including cost.
Hay et al. (2012) motivate evaluating the remaining computation trajectory.
Honda et al. (2024) motivate learned, fixed, and reactive replanning comparisons.
Xu et al. (2026) motivate final-outcome training with a frozen worker.
Our central comparison asks what state adds beyond time in the same decision
round. Report exact rescues and losses alongside intermediate scores.

## Record keeping

Commit code, reproduction commands, and concise milestone findings to saint.
Keep raw traces, weights, and detailed tables local. Existing experiments and
proposed experiments must remain distinguishable.
