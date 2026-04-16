"""Advise-category constraints: prompt-mediated behavior shaping.

A constraint belongs here if its effect is delivered by altering what the
LLM sees on the next turn (new system message, edited tool result,
blocked-notification text, etc). The LLM can, in principle, ignore it.

These are not weaker than enforce-category — they are the only way to push
on behaviors the runtime cannot observe or intercept. But they must be
labeled honestly so callers know the enforcement model.
"""

from dap.constraints.builtin.advise.inject_reminder import InjectReminder
from dap.constraints.builtin.advise.loop_advisory import LoopAdvisory

__all__ = ["InjectReminder", "LoopAdvisory"]
