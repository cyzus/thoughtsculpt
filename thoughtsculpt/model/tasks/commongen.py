# Node
INSTRUCTION = """\
Your Task - Concepts: {}
"""
TASK_DESCRIPTION = """\
# Instruction Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words. The sentence should describe a common scene in daily life, and the concepts should be used in a natural way. 
# # Examples 
# ## Example 1 - Concepts: "dog, frisbee, catch, throw" - Sentence: The dog catches the frisbee when the boy throws it into the air. 
# ## Example 2 - Concepts: "apple, place, tree, pick" - Sentence: A girl picks some apples from a tree and places them into her basket. 
"""

INIT_RESPONSE = TASK_DESCRIPTION + """\
Instruction:
{}
Sentence:
"""

NEW_CANDIDATE = TASK_DESCRIPTION + """\
Instruction:
{instruct}

Here is a proposed sentence.
{solution}

Here is the feedback of outline item.
{feedback}

Based on the feedback, can you make a revised solution?
# Sentence:
"""


NEW_CANDIDATE_COT = TASK_DESCRIPTION + """\
Instruction:
{instruct}

Here is a proposed sentence.
{solution}

Can you make a revised solution?
# Sentence:
"""

EVALUATE_CURRENT = TASK_DESCRIPTION + """\
Instruction:
{}

Here is a proposed sentence.
{}

Rate the sentence based on the following criteria:
1. The sentence covers all the concepts.
2. The sentence describes a common scene in daily life.

Write 1-5 for each criterion, where 1 is the lowest and 5 is the highest, and provide a reason for each score.

# Write in this format:
1. [score: 1-5] [reason]
2. [score: 1-5] [reason]

# Example:
1. [score: 5] [The sentence covers all the concepts.]
2. [score: 2] [The concept "fix" is not used correctly because it doesn't make sense to fix a sausage wrap.]

# Your response:
"""

FEEDBACK = """\
Missing concepts: "{}"
"""

REVISE_SOLUTIONS = TASK_DESCRIPTION + """\
Instruction:
{}

Here is a proposed sentence.
{}

Do you think that the proposed sentence is good enough? Write "no need to improve" if you think 1) the sentence covers all the concepts listed in the instruction; and 2) the sentence describes a common scene in daily life.

Otherwise, write "still need to improve" and provide a reason.

# Write in this format:
[No need to improve/still need to improve] [reason] xxx (50 words max)

# Example 1:
[still need to improve] the sentence misses the concept "dog", "ladder", and "drum".
# Example 2:
[still need to improve] the cat does not fly.

# Your response:
"""


# EVALUATION 
COMPARE = """\
Here are three sentences which are required to use the concepts.
{concepts}

# Sentence 1:
{s1}
# Sentence 2:
{s2}
# Sentence 3:
{s3}

Choose a sentence that you think is the best based on the following criteria:
1. The sentence covers all the concepts.
2. The sentence describes a common scene in daily life.
3. The concepts should be used in a natural way

Write you response in this format:
[1/2/3] [reason] (please include the brackets)

# Your response:
"""
