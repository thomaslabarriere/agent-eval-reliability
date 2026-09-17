"""Pure, dependency-free statistics for agent-eval reliability.

Nothing here does I/O or calls an LLM. Everything is a deterministic function of
its inputs, so a verdict can be recomputed and audited from the recorded run
outcomes rather than trusted from prose.
"""
