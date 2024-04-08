import os
import json
import copy

DATA_PATH = os.path.join(os.path.dirname(__file__), '../..', 'datasets')
# ToT environments
class MiniCrosswordsEnv:
    def __init__(self, file='mini0505.json'):
        self.file = os.path.join(DATA_PATH, 'crosswords', file)

        self.file = json.load(open(self.file))
        self.n = len(self.file)
        self.cache = {}
        self.idx = None
        self.times = 0
        self.prompt_status_cache = {}

    def __len__(self):
        return self.n
    
    def copy_position(self):
        return {"idx":self.idx, 
                             "board":copy.deepcopy(self.board), 
                             "status":copy.deepcopy(self.status), 
                             "steps":self.steps}

    def reset(self, idx, board=None, status=None, steps=None):
        self.idx = idx
        self.data, self.board_gt = self.file[idx]
        self.board = ['_'] * 25
        self.ans = ['_____'] * 10
        self.ans_gt = self.get_ans(self.board_gt)
        self.steps = 0
        self.status = [0] * 10  # 0: unfilled; 1: filled; 2: filled then changed
        if board is not None:
            self.board = board.copy()
            self.ans = self.get_ans(self.board)
        if status is not None:
            self.status = status.copy()
        if steps is not None:
            self.steps = steps
        return self.render()
    

    def prompt_status(self, model):
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(self.ans, self.data, self.status):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = value_prompt.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                res = model([prompt])[0]
                self.prompt_status_cache[prompt] = res
            # print(line)
            # print(res)
            # print()
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
        # print(count)
        return count
    
    def render_gt_board(self):
        s = "GT Board:\n"
        for i in range(5):
            s += ' '.join(self.board_gt[i*5:(i+1)*5]) + '\n'
        return s
    
    def render_board(self):
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(self.board[i*5:(i+1)*5]) + '\n'
        return s

    def render_clues(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + '\n'
        return s
    
    def render_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        return s
    
    def render_gt_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        return s

    def render(self, status=True):
        if status:
            return self.render_board() + '\nUnfilled:\n' + self.render_ans(status=0) + '\nFilled:\n' + self.render_ans(status=1) + '\nChanged:\n' + self.render_ans(status=2)
        else:
            return self.render_board() + '\n' + self.render_ans()
    
    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    def step(self, action, clear_if_conflicted=True):
        self.steps += 1
        action = action.split('\n')[-1]
        action = action.split('. ')
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = action

        if len(word) != 5:
            return 'Invalid! Word should have 5 letters.', 0, False, {}
        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            for j in range(5):
                letter_idx = idx *5 + j
                letter = list(word.upper())[j]
                if self.board[letter_idx] != "_" and self.board[letter_idx] != letter and clear_if_conflicted:
                    # clear that column
                    column_idx = letter_idx % 5
                    self.board[column_idx::5] = ['_'] * 5
                self.board[letter_idx] = letter
                    
            
            # self.board[idx*5:(idx+1)*5] = list(word.upper())
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            for j in range(5):
                letter_idx = idx + j * 5
                letter = list(word.upper())[j]
                if self.board[letter_idx] != "_" and self.board[letter_idx] != letter and clear_if_conflicted:
                    # clear that row
                    row_idx = letter_idx // 5
                    self.board[row_idx*5:(row_idx+1)*5] = ['_'] * 5
                self.board[letter_idx] = letter
            # self.board[idx::5] = list(word.upper())
            idx += 5  # for later status update
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
        
        self.new_ans = self.get_ans(self.board)
        # self.status = [2 if (status == 1 and ans != new_ans) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status = [0 if any(letter == '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status[idx] = 1
        self.ans = self.new_ans
        r_all = (self.board == self.board_gt)
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return self.render(), r_all, (r_all or self.steps >= 20), {'r_letter': r_letter, 'r_word': r_word, 'r_game': r_all}


propose_prompt = '''Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.

{input}

Given the current status, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format like this:

h1. proposed_word (medium)
h2. proposed_word (certain)
...
v1. proposed_word (high) 
...

Use "certain" cautiously and only when you are 100% sure this is the correct word. You can list more then one possible answer for each word.

Possible answers:
'''

propose_change_prompt = '''\
Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.

{input}

The current board has some conflicts. Either the Filled and Unfilled words do not fit or "Filled" items don't match the description. 
Given the current status, find "Filled" items that are likely to be wrong and propose new words to replace it, and the corresponding confidence levels (certain/high/medium/low). 
Use "certain" cautiously and only when you are 100% sure this is the correct word.

Using the format like this:
[motivation] explain the reason to change the word
h1. description [original] original_word [new] new_word (medium)
[motivation] explain the reason to change the word
h2. description [original] original_word [new] new_word (high)
...
[motivation] explain the reason to change the word
v1. description [original] original_word [new] new_word (medium)
...
Possible changes:
'''

specific_change_prompt = """\
You are solving a 5 x 5 mini crossword, where each word should has exactly 5 letters.
{input}    
Can you propose a change to {idx} with confidence level (certain/high/medium/low)?
Use "certain" cautiously and only when you are 100% sure this is the correct word.

Write in the format:

1.
{idx}. [description] [original answer] -> [proposed 5 letter word] (medium)
2.
{idx}. [description] [original answer] -> [proposed 5 letter word] (low)
...
"""

check_prompt = """\
You are solving a 5 x 5 mini crossword, where each word should has exactly 5 letters.
You are checking your already filled-in answers for 5x5 crosswords.
Here are your answers:
{obs}    
Evaluate the answers and assign a confidence level (certain/high/medium/low) to each of them.
Use "certain" cautiously and only when you are 100% sure this is the correct word.

Write your response in the format:
v1. [description] - [original answer] (high)
v2. [description] - [original answer] (medium)
...
h1. [description] - [original answer] (low)
...
"""

value_prompt = '''Evaluate if there exists a five letter word of some meaning that fit some letter constraints (sure/maybe/impossible).

Incorrect; to injure: w _ o _ g
The letter constraint is: 5 letters, letter 1 is w, letter 3 is o, letter 5 is g.
Some possible words that mean "Incorrect; to injure":
wrong (w r o n g): 5 letters, letter 1 is w, letter 3 is o, letter 5 is g. fit!
sure

A person with an all-consuming enthusiasm, such as for computers or anime: _ _ _ _ u
The letter constraint is: 5 letters, letter 5 is u.
Some possible words that mean "A person with an all-consuming enthusiasm, such as for computers or anime":
geek (g e e k): 4 letters, not 5
otaku (o t a k u): 5 letters, letter 5 is u
sure

Dewy; roscid: r _ _ _ l
The letter constraint is: 5 letters, letter 1 is r, letter 5 is l.
Some possible words that mean "Dewy; roscid":
moist (m o i s t): 5 letters, letter 1 is m, not r
humid (h u m i d): 5 letters, letter 1 is h, not r
I cannot think of any words now. Only 2 letters are constrained, it is still likely
maybe

A woodland: _ l _ d e
The letter constraint is: 5 letters, letter 2 is l, letter 4 is d, letter 5 is e.
Some possible words that mean "A woodland":
forest (f o r e s t): 6 letters, not 5
woods (w o o d s): 5 letters, letter 2 is o, not l
grove (g r o v e): 5 letters, letter 2 is r, not l
I cannot think of any words now. 3 letters are constrained, and _ l _ d e seems a common pattern
maybe

An inn: _ d _ w f
The letter constraint is: 5 letters, letter 2 is d, letter 4 is w, letter 5 is f.
Some possible words that mean "An inn":
hotel (h o t e l): 5 letters, letter 2 is o, not d
lodge (l o d g e): 5 letters, letter 2 is o, not d
I cannot think of any words now. 3 letters are constrained, and it is extremely unlikely to have a word with pattern _ d _ w f to mean "An inn"
impossible

Chance; a parasitic worm; a fish: w r a k _
The letter constraint is: 5 letters, letter 1 is w, letter 2 is r, letter 3 is a, letter 4 is k.
Some possible words that mean "Chance; a parasitic worm; a fish":
fluke (f l u k e): 5 letters, letter 1 is f, not w
I cannot think of any words now. 4 letters are constrained, and it is extremely unlikely to have a word with pattern w r a k _ to mean "Chance; a parasitic worm; a fish"
impossible

{input}
'''

# Node
TASK_DESCRIPTION = """\
Task Description:
Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters. Your goal is to fill in the crossword with words based on the hints provided.
"""

NEW_CANDIDATE = TASK_DESCRIPTION + """\
#Current board:
{obs}

#Strategy:
{feedback}

Given the current status of the board and the strategy, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format like this:
Use "certain" cautiously and only when you are 100% sure this is the correct word. You can list more then one possible answer for each word.

h1. [hint: _____] xxxxx (medium)
h2. [hint: _____] xxxxx (certain)
...
v1. [hint: _____] xxxxx (high) 
...

#Answer like the examples suggest:
h1. [A financial loss; a negative profit; to remove bits from: D_B__] DEBTS (low)
h2. [Fatuous; empty headed: _____] INANE (high)
...
v1. [A dice player; something that cuts into small cubes: _____] DICER (high)
v5. [An Indian tent: _____] TEPEE (medium)

Each line can only have one candidate answer. 
#Your response:
"""


NEW_CANDIDATE_COT = TASK_DESCRIPTION + """\
#Current board:
{obs}

Given the current status of the board, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format like this:
Use "certain" cautiously and only when you are 100% sure this is the correct word. You can list more then one possible answer for each word.

h1. [hint: _____] xxxxx (medium)
h2. [hint: _____] xxxxx (certain)
...
v1. [hint: _____] xxxxx (high) 
...

#Answer like the examples suggest:
h1. [A financial loss; a negative profit; to remove bits from: D_B__] DEBTS (low)
h2. [Fatuous; empty headed: _____] INANE (high)
...
v1. [A dice player; something that cuts into small cubes: _____] DICER (high)
v5. [An Indian tent: _____] TEPEE (medium)

Each line can only have one candidate answer. 
#Your response:
"""


REVISE_SOLUTIONS = TASK_DESCRIPTION + """\
Current board:
{obs}
Evaluate the current board and provide a strategy on how to continue to fill in the blank or correct potential mistakes.
Write your response in the format:
v1. [reasoning and potential answers] 
v2. [reasoning and potential answers] 
...
h1. [reasoning and potential answers]
...
Example:
v2. [Current answer: tough; since  the filled in h1. is debit; e is conflicted with t, we could consider other options such as ENURE]
v3. [Current answer: ??? CUTUP could be a potential answer]
Your response:
"""
# REVISE_SOLUTIONS = TASK_DESCRIPTION + """\
# Current board:
# {obs}
# Evaluate the current board and provide a strategy on how to continue to fill in the blank or correct potential mistakes.
# Your response:
# """



EVALUATE_CURRENT = TASK_DESCRIPTION + """\
Current board:
{obs}

How successfully do you think the crosswords had been filled out so far? 
Gives a score of 100 if the crosswords have been solved; 
A score of 0 if the crosswords haven't been filled at all.


"""