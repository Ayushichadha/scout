"""Plot paired mission outcomes from the synthetic tool-revision pilot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

trace = pd.read_csv("outputs/tool-revision-pilot/test_trace.csv.gz")
pairs = trace.pivot(index=["regime", "episode"], columns="policy", values="solved")
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), sharey=True, sharex=True)
for ax, (policy, title) in zip(
    axes,
    [
        ("validation_clock", "Compared with fixed timing"),
        ("failure_streak", "Compared with failure streak"),
    ],
):
    change = pairs.state_supervisor - pairs[policy]
    for i, regime in enumerate(["persistent", "transient", "stable"]):
        group = change.xs(regime)
        lost, rescued = (group < 0).sum(), (group > 0).sum()
        ax.barh(i, -lost, color="#b45646")
        ax.barh(i, rescued, color="#247069")
        ax.text(-lost - 1, i, str(lost), va="center", ha="right")
        ax.text(rescued + 1, i, str(rescued), va="center")
    ax.axvline(0, color="#333", lw=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lost   ←   workflows   →   rescued")
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].set_yticks(
    range(3), ["Persistent outage", "Transient outage", "Stable service"]
)
axes[0].set_xlim(-45, 60)
fig.suptitle("When does a state-aware tool switch change mission success?")
fig.tight_layout()
out = Path("papers/trust4rsi/agent_revision_outcomes.pdf")
fig.savefig(out, bbox_inches="tight")
