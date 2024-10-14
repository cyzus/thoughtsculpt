TASK_DESCRIPTION = """\
Use the given four numbers and basic arithmetic operations (+ - * /) to obtain 24. 
You can use the numbers only once but you can use them in any order.
"""


SOLUTION_OUTPUT_FORMAT = """\
# Think step by step first. Then, please output the solution in the following format (in a python code block)
```python
(1 + 2) * (2 * 4)
```
# Your response. 
"""



REVISE_SOLUTIONS = """\
# Instruction
{instruction}

# Current Solution
{solution}

Calculate the result of the current solution.
Do you think the solution is correct? If not, please provide feedback. 

# Output format

```json
{{
    "calculation": "step by step calculation of the current solution",
    "result: int,
    "feedback": "Your feedback here",
    "correct" : true/false
}}
```
"""
