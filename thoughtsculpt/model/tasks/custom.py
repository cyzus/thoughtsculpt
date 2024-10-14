PROPOSE_PROMPT = """
# Instruction
{instruction}

# Current Solution
{solution}

# Feedback for the solution
{feedback}

Propose a new solution:
"""

INIT_RESPONSE = """\
# Instruction
{instruction}
"""


REVISE_SOLUTIONS = """\
# Instruction
{instruction}

# Current Solution
{solution}

Do you think the solution is correct? If not, please provide a feedback. Think step by step.

# Output format

```json
{{
    "feedback" : "Your feedback here",
    "correct" : true/false
}}
```
"""
