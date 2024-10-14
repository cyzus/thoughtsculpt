from thoughtsculpt.model.node import Node, CustomNode
from thoughtsculpt.model.tasks.game24 import (
    TASK_DESCRIPTION, SOLUTION_OUTPUT_FORMAT, REVISE_SOLUTIONS
)
import re

class Node24(CustomNode):
    task_description = TASK_DESCRIPTION
    solution_output_format = SOLUTION_OUTPUT_FORMAT
    revise_prompt = REVISE_SOLUTIONS

    def __init__(self, *args, **kwargs):
        super(Node24, self).__init__(*args, **kwargs)


    def parse_solution(self, output):
        # parse python block code
        # ```python```
        pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        python_blocks = pattern.findall(output)
        if len(python_blocks) == 0:
            return ""
        return python_blocks[0]
    
    def is_terminal(self):
        if self.feedback is None:
            self.revise_answers()
        return super().is_terminal()
    
    def reward(self):
        if self.feedback is None:
            self.revise_answers()
        if "result" not in self.feedback:
            return 0
        else:
            try:
                result = int(self.feedback["result"])
                return 1/(1+abs(24-result))*10
            except:
                print(f"Error: result is not an integer: {self.feedback['result']}")
                return super().reward()

    def get_feedback(self):
        existing_candidates = []
        parent = self.parent
        while parent is not None:
            existing_candidates.append(parent.position)
            parent = parent.parent
        existing_candidates_text = "\n".join(existing_candidates[::-1])
        calculation = self.feedback.get("calculation", "")
        result = self.feedback.get("result", "")
        feedback = self.feedback.get("feedback", "")
        overall_feedback = f"Wrong solutions: {existing_candidates_text}\n Calculation: {calculation}\nResult: {result}\nFeedback: {feedback}"
        return overall_feedback