"""Hybrid-category constraints: advise first, enforce on escalation.

A constraint belongs here if it deliberately combines both mechanisms in a
single policy — e.g. "nudge the LLM with a reminder for N steps, then if the
behavior persists, abort the run". The category exists to keep the advise
and enforce submodules each honest about being one-shot in mechanism,
without forcing authors to choose when a real policy wants both.

Empty for now. Candidates: LoopDetector (N advises → abort), BudgetWarner
(warn via prompt when 80% of tool budget consumed → abort at 100%).
"""

__all__: list[str] = []
