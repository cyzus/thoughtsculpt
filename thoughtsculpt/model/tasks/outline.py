PROMPT = """\
{}
Given the story outlines above, do you think that the new story point below is interesting?
{}
"""

MASKED_PROMPT = """\
Here is a story outline.
{}
Do you think that the outline item {} is interesting?
The interesting outline items would engage readers to read the story. Otherwise, it's boring and should be revised.
"""

MASKED_OUTLINE_PROMPT = """\
{}
Here are a story outline with some blank items:
{}

Write in this format:
1.
2.
3.
...

Remember your role, and fill up the missing outlines. Make sure that the story is coherent. The output should have the same number of outline items as the original one.
"""

INTERESTING = """\
You are a popular novel writer. You are now making an outline for the story. You know how to engage with the readers by not limited to introducing interesting characters and unexpected twist.
"""

NON_INTERESTING = """\
You are a bad novel writer. You are now making an outline for the story. Your outlines are boring; you don't know how to engage with the readers. Readers won't find your story interesting at all.
"""

EVALUATE_ITEMS = """\
Here is a story outline.
{}
Which continuous {} outlines items do you think are least interesting?
The interesting outline items should engage readers to read the story. Otherwise, it's boring and should be revised. The interesting level would be from 1 to 5, where 1 is the least interesting and 5 is the most interesting.

Write in this format:
[reason] [start_index]-[end_index] [interesting level: 1-5]

Example:
[reason: too repetitive] [9]-[10] [interesting level: 2]
"""

EVALUATE_ITEMS = """\
Here is a story outline.
{}
Which continuous {} outlines items do you think are least interesting?
The interesting outline items should engage readers to read the story. Otherwise, it's boring and should be revised. The interesting level would be from 1 to 5, where 1 is the least interesting and 5 is the most interesting.

Write in this format:
[reason: too repetitive/cliche plot/unsurprising/etc] [start_index]-[end_index] [interesting level: 1-5]

Example:
[reason: too repetitive] [2]-[3] [interesting level: 2]

Your response:
"""


# Node
TASK_DESCRIPTION = """\
# Task Description
You are a popular novel writer. You are now making an interesting outline for the story. 
You know how to engage with the readers by not limited to introducing interesting characters and unexpected twist.
You also know how to make the story outline coherent and consistent.
"""

NEW_CANDIDATE = TASK_DESCRIPTION + """\
# Original Outline
{outline}

# Feedback
{feedback}

Based on the feedback and the task description, can you make a better story outline by replacing the items suggested by the feedback? Think step by step.

Write the outline in this format just like the original outline from [1] to [{num}]:
Thought process: 
...
Outline:
[1] ...
[2] ...
...

# Your response:
"""

NEW_CANDIDATE_COT = TASK_DESCRIPTION + """\
# Original Outline
{outline}

Based on the task description, can you make a better story outline? Think step by step.

Write the outline in this format just like the original outline from [1] to [{num}]:
Thought process: 
...
Outline:
[1] ...
[2] ...
...

# Your response:
"""


EVALUATE_CURRENT = TASK_DESCRIPTION + """\
# Original Outline
{}

Do you think that this outline is good enough.
Write a score from 1 to 100 where 100 means the outline is perfect based on the task description, and provide an explanation on strengths and weaknesses. Please be specific.
# Write in this format:
[score: 1-100] [reason] xxx (50 words max)

# Example:
[score: 50] [reason] the current outline is too predictable

# Your response:

"""

REVISE_SOLUTIONS = TASK_DESCRIPTION + """\
Here is a story outline.
{}
Which continuous {} outlines items do you think are least interesting?
The interesting outline items should engage readers to read the story. Otherwise, it's boring and should be revised. The interesting level would be from 1 to 5, where 1 is the least interesting and 5 is the most interesting.

Write in this format:
Thought Process:
...
[reason: too repetitive/cliche plot/unsurprising/etc] [start_index]-[end_index] [interesting level: 1-10]

Example:
Thought Process:
Outline items 9 and 10 talks about the same thing over outline items 7 and 8. It's too repetitive.
[reason: too repetitive] [9]-[10] [interesting level: 5]

Can you provide {} proposals? 

# Your response:

"""

# Compare

COMPARE = """\
Which story outline is more interesting and more coherent?
Outline A:
{}

Outline B:
{}

Pick your answer from ['Outline A', 'Outline B', 'both', 'neither'].  
Generate a short explanation for your choice first. The better outline is A OR The better outline is B OR The outlines are equally good OR Neither outline is good.

Format: <explanation> <choice> STOP
"""
