# Trust4RSI revision experiments

This work extends Scout's feudal subgoals with learned persistence and revision.
The earlier paper and its historical artifacts remain under papers/beyond-the-clock.

Generated reports, traces, figures, and checkpoints stay local. This directory
documents the reproducible code entry points.

From the repository root:

    python -m scripts.replay_supervisors
    python scripts/run_tool_revision_pilot.py --output /tmp/scout-tool
    python scripts/run_optional_tool_revision.py --output /tmp/scout-optional
    python scripts/run_feudal_task_pilot.py --help

Official TRM source is an external dependency. Retrieve and pin it before running
the CPU reference check:

    git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git /tmp/scout-reference-trm
    git -C /tmp/scout-reference-trm checkout c01103738605ba39d1430519b1ee0c62f4c707f8
    python scripts/check_reference_trm.py --source /tmp/scout-reference-trm

The reference check uses random weights and synthetic tokens. The feudal task pilot
uses a shared-core prototype. Task-trained reference TRM with Scout remains pending.
