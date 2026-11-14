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

## Methods informed by the four reviewer references

This section records the agreed research direction developed in the discussion.
The experimental choices below are our proposed adaptations of the cited work.
Implementation and empirical validation remain pending where explicitly stated.

### Decision value

For a frozen worker, define the value of refreshing at state s and pass t as
the difference in expected terminal success between refresh and persist, under
the same remaining budget and subsequent policy. Both branches must start from
the same complete state. Repeated stochastic continuations estimate expectations;
one branch pair provides a noisy outcome sample.

Russell and Wefald's metareasoning framing motivates evaluating computation by
its downstream decision utility. Our primary comparison uses matched budgets.
Report actual worker computation and controller overhead separately. A secondary
cost-sensitive objective may subtract a preregistered cost coefficient from
success, with coefficient selection confined to development data.
[Principles of Metareasoning](https://doi.org/10.1016/0004-3702(91)90015-C).

Hay et al. formulate computation selection through expected improvements in
decision quality. We use that framing to motivate terminal branch evaluation
and an explicit continuation policy. A single-pass loss difference is a secondary
measurement. Their theoretical results concern their selection setting; applying
this framing to learned goal refresh requires empirical validation.
[Selecting Computations](https://arxiv.org/abs/1207.5879).

### Comparisons required by the research question

Honda et al. show that replanning strategy effectiveness depends on environment
and planner configuration. Our adaptation compares always-persist, a
validation-selected fixed revision schedule, a validation-selected reactive rule,
and learned revision. HRM reactive features must use observable prediction or
state change without target labels. SQLite rules may use observed failure
streaks. Subgroup reporting should expose gains and losses across task conditions.
[When to Replan?](https://arxiv.org/abs/2304.12046).

Xu et al. freeze the base VLA and learn sequential continue/replan decisions from
trajectory outcomes, including replanning efficiency in the reward. Their
Bernoulli continuation formulation motivates a sequential binary supervisor
and endpoint-based training. First fit matched time-only and time-plus-state
supervisors to the same paired branch data. A subsequent trajectory-trained
Bernoulli baseline is a proposed adaptation. Claiming a BCP reproduction requires
matching and verifying its actual training algorithm and reward.
[Continue or Replan?](https://arxiv.org/abs/2608.03483).

Match action availability, remaining budget, fitting data, model capacity, and
tuning effort across the learned supervisors. Report within-pass comparisons,
then evaluate frozen policies from episode start on untouched source problems.
Use source-level paired intervals and separate rescue and loss counts.

### Application to the implemented systems

In HRM, persist retains the latent goal and refresh recomputes it from the current
reasoning state. In the SQLite bridge, persist retains the cached manager goal
used by the service-selection worker and refresh recomputes that goal from
observations. SQLite's endpoint is successful record transfer under its call
budget. Its synthetic faults and small learned worker bound the transfer claim.
A general coding agent would introduce an explicit textual plan intervention.

The primary contribution under investigation is the incremental value of state
for selecting beneficial revisions beyond timing and reactive behavior.
Adaptive timing and frozen-worker supervision already have direct precedents.

### Evidence status and next decision

The broader HRM replay found zero exact solves on 80 calibration-source examples.
The 1,600-update repeated-batch run has completed at 96.72% token accuracy and
0/4 exact solves, matching the 400-update endpoint. Further training-budget
increases need an explanation for this plateau. Saved weights and predictions
now support inspection without repeating training.

For the existing SQLite bridge, the next missing measurement is the terminal
persist/refresh disagreement by decision step, followed by a matched learned
time-only baseline. This uses the implemented software task and tests whether
the current fixed-clock comparison hides a temporal learning advantage.

## November 20 submission priority

Following the author's November 14 decision, the existing SQLite goal-refresh
study is the main experiment. HRM/ARC supplies the motivating diagnostic.
Prioritize matched learned time-only and state-aware decisions, endpoint branch
outcomes, fresh-workflow replication, and manuscript preparation.

The SQLite runner now includes a train-fitted time-only optimal-stopping policy
using the same paired outcomes and backwards decision order as the state model.
At each time its constant predictor is the mean advantage, equivalent to fitting
the unpenalized intercept in the state model. Full trajectory evaluation includes
this policy, and reports persist/refresh outcomes by pass.

The first fresh seed (2026111401, 500 evaluation workflows) scored 80.2% for
both learned policies and 80.4% for reactive control. State control rescued 17
and lost 4 workflows against persistence, but had identical endpoint outcomes
to time-only control. This is one development result. Goal revision can change
task completion in this setting; added state has not demonstrated extra value.
