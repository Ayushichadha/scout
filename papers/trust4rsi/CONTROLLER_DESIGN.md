# Controller design for the SQLite experiment

## Three components

The worker observes tool responses and decides whether to switch services.
A manager produces a normalized 16-dimensional goal and a gate. The worker
uses its encoded observation plus the gated cached goal to predict the value
of switching services. The supervisor decides when to refresh that goal.

The worker and manager are trained together on training workflows, then frozen.
The supervisor is fitted afterward. This is the SQLite implementation of the
persistent-goal idea. HRM uses a recurrent reasoning worker with a different
state representation.

## State available before the decision

At call k, observation includes an intercept, latest response, success fraction
over the latest two calls, cumulative success fraction, remaining required
records divided by twelve, total required records divided by twelve, and k/12.
The supervisor also receives the distance between a newly proposed goal and
the cached goal. Each call position has its own fitted coefficient vector.
Final outcomes and future tool responses are excluded from inference inputs.

Computing the candidate goal is already required by the current state features.
Thus this implementation tests the effect of adopting a refreshed goal. Any
claim of saving manager computation requires separately measuring or changing
that candidate-goal computation.

## Actions and budget

Persist keeps the cached goal. Refresh replaces the goal and gate using current
observations. Each workflow has twelve tool calls, at most one service switch,
and at most one optional goal refresh at calls 3–10. A service switch consumes
the environment's restoration cost. Manager inference and supervisor overhead
are additional computation beyond the matched tool-call budget.

## Training target

For each training workflow, evaluate completion under every allowed fixed
refresh time and under persistence. Fit the supervisor backwards through time.
At call k its target is completion when refreshing now minus completion under
the already-fitted future revision policy. Fit ridge regression with penalty
10 and an unpenalized intercept. Refresh when predicted advantage is positive;
otherwise continue. The worker keeps making service decisions during this wait.

The time-only baseline follows the same backwards construction using a constant
predictor at each call. Its prediction is mean training advantage at that time.
This tests whether observations add value beyond a learned temporal schedule.
The state model has additional coefficients, so capacity is not identical.

## Evaluation and interpretation

Compare whole trajectories with persistence, validation-selected fixed refresh,
learned time-only refresh, and learned state-aware refresh. A reactive service
switch rule is a system-level reference with a different action interface.
Report endpoint rescues and losses, paired intervals, and refresh frequency.

The hindsight oracle chooses the best refresh time separately for each
evaluation workflow using its outcomes. It measures available opportunity and
is an upper bound over these interventions, not an executable supervisor.
Synthetic service faults and the small worker define the scope of the results.

## Reviewer connections

Russell and Wefald motivate final utility and computation cost. Hay et al.
motivate the value of computation for later decisions. Honda et al. motivate
fixed, reactive, and learned replanning comparisons. Xu et al. motivate
trajectory-outcome training while freezing the worker. Our ridge supervisor
is a proposed adaptation of these principles, not a reproduction of BCP.
See RESEARCH_DIRECTION.md for references and experiment status.
