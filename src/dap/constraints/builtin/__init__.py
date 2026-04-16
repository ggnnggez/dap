"""Built-in constraints, grouped by enforcement mechanism.

Import from the category that matches the constraint's enforcement model.
The category is deliberately visible at the call site so the reader can
tell, without opening the class, whether the LLM can ignore the constraint
or not.

  dap.constraints.builtin.enforce  — runtime-level enforcement. The LLM
                                     cannot "retry through" these.
  dap.constraints.builtin.advise   — prompt-mediated. The LLM may comply
                                     or ignore.
  dap.constraints.builtin.hybrid   — combine advise escalation with an
                                     enforce fallback.
"""
