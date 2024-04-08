import re
from nltk.metrics import edit_distance
import numpy as np

class Node():
    """A node class for Pathfinding"""

    def __init__(self, parent=None, position=None, **kargs):
        self.parent = parent
        self.position = position
        self.f = None
        self.h = None
        self.g = None

    def can_end(self):
        return False
    
    def get_candidates(self, num_candidates):
        return [Node()] * num_candidates
    
    def get_random_candidate(self):
        return self.get_candidates(num_candidates=1)
    
    def get_cot_candidate(self):
        return [Node()]

    def get_tot_candidates(self, num_candidates=3):
        candidates = []
        for i in range(num_candidates):
            candidate = self.get_cot_candidate()
            if candidate == self:
                continue
            candidates.append(candidate)
        return candidates


    def __eq__(self, other):
        return self.position == other.position
    
    def get_future_score(self):
        return 1
    
    def get_current_score(self):
        return 1
    
    def reward(self):
        if not self.f:
            self.compute_scores()
        return self.f
    
    def __hash__(self):
        return hash(str(self.position))
    
    def compute_scores(self,):
        self.g = self.get_current_score()
        self.h = self.get_future_score()
        self.f = self.g + self.h


class ContentNode(Node):
    def __init__(self, model, 
                 external_model=None, parent=None, position=None, **kargs):
        self.parent = parent
        self.position = position
        self.model = model
        self.external_model = external_model
        self.f, self.h, self.g = None, None, None
        
    
    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __str__(self):
        return self.position_to_text() + "\n"
    
    def __hash__(self):
        return hash(self.__str__())
    

    def similarity(self, other):
        a = self.position_to_text()
        b = other.position_to_text()
        similarity =  1 - edit_distance(a, b) / len(a)
        return similarity

    
    def similar_to(self, other):
        similarity =  self.similarity(other)
        return similarity > 0.6

    
    def compute_scores(self, **kargs):
        self.g, reason = self.get_current_score()
        self.h = 0
        self.f = self.g
    
    def is_terminal(self):
        if self.f is None:
            self.compute_scores()
        return self.f > 95
    
    def get_random_candidate(self):
        return self.get_candidates(num_candidates=1)[0]

    
    def get_cot_candidate(self):
        from thoughtsculpt.model.tasks.outline import NEW_CANDIDATE_COT
        outline_text = self.position_to_text()
        prompt = NEW_CANDIDATE_COT.format(outline=outline_text, num=len(self.position))
        candidate_output = self.model([prompt])[0]
        position = self.parse_candidate(candidate_output)
        candidate = ContentNode(parent=self, position=position,
                                     model=self.model, external_model=self.external_model)
        return candidate
    
    
        
    
    def get_candidates(self, num_candidates=3):
        from thoughtsculpt.model.tasks.outline import NEW_CANDIDATE
        feedbacks = self.revise_answers(num_candidates=num_candidates, num_items=2)
        # print(feedbacks)
        outline_text = self.position_to_text()
        candidates = []
        for feedback in feedbacks.split("\n")[:num_candidates]:
            prompt = NEW_CANDIDATE.format(outline=outline_text, feedback=feedback, num=len(self.position))
            candidate_output = self.model([prompt])[0]
            position = self.parse_candidate(candidate_output)
            candidate = ContentNode(parent=self, position=position,
                                     model=self.model, external_model=self.external_model)
            if candidate == self:
                continue
            candidates.append(candidate)
        return candidates

    
    def revise_answers(self, num_candidates, num_items=2):
        from thoughtsculpt.model.tasks.outline import REVISE_SOLUTIONS
        outline_text = self.position_to_text()
        prompt = REVISE_SOLUTIONS.format(outline_text, num_items, num_candidates) 
        output = self.model([prompt])[0]
        return output

    
    def get_future_score(self, reason):
        return 1
    
    def get_current_score(self): # Use an independent evaluator or GPT 
        if self.external_model is None:
            from thoughtsculpt.model.tasks.outline import EVALUATE_CURRENT
            outlines_text = self.position_to_text()
            prompt = EVALUATE_CURRENT.format(outlines_text)
            output = self.model([prompt])[0]
            score, reason = ContentNode.parse_score_reason(output, prompt)
            return score, reason
        else:
            return self.external_model.predict_interestingness(self.position), ""
            
    
    def position_to_text(self):
        return "\n".join([f"[{i+1}] {outline}"  for i, outline in enumerate(self.position)])
    
    
    def parse_candidate(self, candidate_output):
        outlines = re.findall(r'\n?\[\d+\] (.*)', candidate_output)[:len((self.position))]
        if len(outlines) < len(self.position):
            print(f"The number of outlines in the candidate is less than the original ({len(outlines)} < {len(self.position)})")
        return outlines
    
    @staticmethod
    def parse_score_reason(evaluate_output, prompt=None):
        outputlines = evaluate_output.split("\n")
        for line in outputlines:
            
            pattern1 = r'.*\[score: (\d+)\] \[reason\] (.*)'
            matched1 = re.match(pattern1, line)
            pattern2 = r'.*\[score: (\d+)\] (.*)'
            matched2 = re.match(pattern2, line)
            if matched1:
                score, reason = matched1.group(1), matched1.group(2)
                return int(score), reason
            if matched2:
                score, reason = matched2.group(1), matched2.group(2)
                return int(score), reason
        raise ValueError("output:\n {} \ncannot be recognized".format(evaluate_output))
        
    
    @staticmethod
    def parse_score(evaluate_output):
        pattern = r'.*\[score: (\d+)\].*$'
        matched = re.match(pattern, evaluate_output)
        score = matched.group(1)
        return int(score)
    

class CrosswordNode(Node):
    confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  
    def __init__(self, model, env, external_model=None, parent=None, position=None, **kargs):
        self.parent = parent
        self.env = env
        if position is None:
            self.position = self.env.copy_position()
        else:
            self.position = position
        self.model = model
        self.external_model = external_model
        self.f, self.h, self.g = None, None, None

    def __eq__(self, other):
        return self.__str__() == other.__str__()
    
    def __str__(self):
        s = ""
        for i in range(5):
            s += ''.join(self.position["board"][i*5:(i+1)*5]) + '\n'
        return s
    
    def __hash__(self):
        return hash(self.__str__())

    
    def is_terminal(self):
        env = self.env
        env.reset(**self.position)
        r_all = (env.board == env.board_gt)
        num_filled_words = np.count_nonzero(self.position["status"])
        return r_all or num_filled_words == 10

    def can_end(self):
        return self.is_terminal()
    
    def compute_scores(self, **kargs):
        self.g, reason = self.get_current_score()
        self.h = self.get_future_score(reason)
        self.f = (self.h + self.g)/2
        
    def revise_answers(self):
        from thoughtsculpt.model.tasks.crosswords import REVISE_SOLUTIONS
        obs = self.env.render()
        prompt = REVISE_SOLUTIONS.format(obs=obs)
        output = self.model([prompt])[0]
        return output

    def get_tot_candidates(self, num_candidates=3):
        from thoughtsculpt.model.tasks.crosswords import NEW_CANDIDATE_COT
        candidates = []
        count = 0
        while not candidates and count < 3:
            count += 1
            env = self.env
            env.reset(**self.position)
            obs = env.render()
            prompt = NEW_CANDIDATE_COT.format(obs=obs)
            answers = self.model([prompt])[0]
            parsed_response = CrosswordNode.parse_response(answers)
            candidate_scores = {}
            for action, score in parsed_response:
                candidate_scores[action] = candidate_scores.get(action, 0) + score
            sorted_candidates = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
            
            for action in sorted_candidates[:num_candidates]:
                env.step(action)
                new_position = env.copy_position()
                candidate = CrosswordNode(model=self.model, env=env, parent=self, position=new_position)
                env.reset(**self.position)
                candidates.append(candidate)
        if not candidates:
            raise ValueError("Couldn't find candidates")
        return candidates
    
    def get_candidates(self, num_candidates=5):
        from thoughtsculpt.model.tasks.crosswords import NEW_CANDIDATE
        candidates = []
        count = 0
        while not candidates and count < 3:
            count += 1
            env = self.env
            env.reset(**self.position)
            obs = env.render()
            print(obs)
            feedback = self.revise_answers()
            print(feedback)
            prompt = NEW_CANDIDATE.format(obs=obs, feedback=feedback)
            answers = self.model([prompt])[0]
            parsed_response = CrosswordNode.parse_response(answers)
            candidate_scores = {}
            for action, score in parsed_response:
                candidate_scores[action] = candidate_scores.get(action, 0) + score
            sorted_candidates = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
            
            for action in sorted_candidates[:num_candidates]:
                env.step(action)
                new_position = env.copy_position()
                candidate = CrosswordNode(model=self.model, env=env, parent=self, position=new_position)
                env.reset(**self.position)
                candidates.append(candidate)
        if not candidates:
            raise ValueError("Couldn't find candidates")
        return candidates

            
    
    def get_future_score(self, reason):
        count = self.env.prompt_status(self.model)
        score = count["impossible"] * 0.1 / 10 + count["maybe"] * 0.5 / 10 + count["sure"] / 10
        return score / 3
    
    def get_current_score(self):
        num_filled_words = np.count_nonzero(self.position["status"])
        word_score = num_filled_words/10
        num_filled_letters = 25 - self.__str__().count("_")
        letter_score = num_filled_letters/25
        return (word_score + letter_score)/2, ""
    
    @staticmethod
    def parse_response(response):
        # split the response into lines
        lines = response.split('\n')
        parsed_lines = [CrosswordNode.parse_line(line) for line in lines]
        parsed_lines = [line for line in parsed_lines if line is not None]
        parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), CrosswordNode.confidence_to_value.get(line[2], 0)) for line in parsed_lines]
        return parsed_lines
    
    @staticmethod
    def parse_line(input_str):
        pattern1 = r'^([hv][1-5])\. .* ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'
        # use regex to extract the parts of the input string
        match = re.match(pattern1, input_str)

        if match:
            # extract the matched groups
            parts = [match.group(1), match.group(2), match.group(3)]
            return parts

class CommonNode(Node):
    def __init__(self, model, concepts, parent=None, position=None):
        from thoughtsculpt.model.tasks.commongen import INSTRUCTION
        self.model = model
        self.concepts = concepts
        self.concepts_str = ", ".join(concepts)
        self.instruction = INSTRUCTION.format(self.concepts_str)
        self.parent = parent
        if position is None:
            self.position = self.get_init_response()
        else:
            self.position = position
        self.f, self.h, self.g = None, None, None
        
    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __str__(self):
        return self.position
    
    def __hash__(self):
        return hash(self.__str__())
    
    def get_init_response(self):
        from thoughtsculpt.model.tasks.commongen import INIT_RESPONSE
        prompt = INIT_RESPONSE.format(self.instruction)
        output = self.model([prompt])[0]
        return output
        
    def get_current_score(self):
        from thoughtsculpt.model.tasks.commongen import EVALUATE_CURRENT
        prompt = EVALUATE_CURRENT.format(self.instruction, self.position)
        output = self.model([prompt])[0]
        score = CommonNode.parse_score(output)
        return score

    def get_future_score(self):
        count = len([c for c in self.concepts if c in self.position])
        return count/len(self.concepts)
        
        
    
    def compute_scores(self):
        self.g = self.get_current_score()
        self.h = self.get_future_score()
        self.f = (self.g + self.h)/2
        
    
    def can_end(self):
        if self.h is None:
            self.compute_scores()
        return self.h == 1
    
    def is_terminal(self):
        if self.h is None:
            self.compute_scores()
        return self.h == 1
    
    def revise_answer(self):
        from thoughtsculpt.model.tasks.commongen import REVISE_SOLUTIONS
        
        prompt = REVISE_SOLUTIONS.format(self.instruction, self.position)
        output = self.model([prompt])[0]
        feedback = CommonNode.parse_revision(output)
        return feedback
        
    def get_cot_candidate(self):
        from thoughtsculpt.model.tasks.commongen import NEW_CANDIDATE_COT
        prompt = NEW_CANDIDATE_COT.format(instruct=self.instruction, solution=self.position)
        output = self.model([prompt])[0]
        candidate = CommonNode(model=self.model, concepts=self.concepts, parent=self, position=CommonNode.parse_sentence(output))
        return candidate

    def get_candidates(self, num_candidates=3):
        from thoughtsculpt.model.tasks.commongen import NEW_CANDIDATE, FEEDBACK
        candidates = []
        feedback = self.revise_answer()
        
        if feedback == 0 and self.h == 1:
            return candidates
        prompt = NEW_CANDIDATE.format(instruct=self.instruction, solution=self.position, feedback=feedback)
        outputs = self.model([prompt]*num_candidates)
        for output in outputs:
            candidate = CommonNode(model=self.model, concepts=self.concepts, parent=self, position=CommonNode.parse_sentence(output))
            candidates.append(candidate)
        return candidates           
    
    @staticmethod
    def parse_revision(response):
        response = response.lower()
        if "no need to improve" in response:
            return 0
        else:
            return response
        
    @staticmethod
    def parse_sentence(response):
        if "Sentence: " in response:
            return response.split("Sentence: ")[1]
        else:
            return response
    
    @staticmethod
    def parse_score(evaluate_output):
        lines = evaluate_output.split("\n")
        patterns = [
            r'.*1. \[score: (\d+)\].*$',
            r'.*2. \[score: (\d+)\].*$',
        ]
        scores = [0, 0]
        for line in lines:
            for i, pattern in enumerate(patterns):
                matched = re.match(pattern, line)
                if matched:
                    scores[i] = int(matched.group(1))
        
        score = (scores[0] * 0.9 + scores[1] * 0.1)/5            
        return score
        
        
        
        
        
        
        
    
