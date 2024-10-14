from nltk.metrics import edit_distance
import re
import numpy as np
import json
from thoughtsculpt.model.tasks.custom import PROPOSE_PROMPT, INIT_RESPONSE, REVISE_SOLUTIONS

class CustomNode():
    task_description = ""
    initial_prompt = INIT_RESPONSE
    revise_prompt = REVISE_SOLUTIONS
    base_propose_prompt = PROPOSE_PROMPT
    solution_output_format = ""


    def __init__(self, parent=None, instruction=None, position=None, model=None, **kargs):
        self.parent = parent
        self.position = position
        self.instruction = instruction
        self.model = model
        self.propose_prompt = self.task_description + self.base_propose_prompt + self.solution_output_format
        self.initial_prompt = self.task_description + self.initial_prompt + self.solution_output_format
        self.revise_prompt = self.task_description + self.revise_prompt
        self.f = None
        self.h = None
        self.g = None
        self.feedback = None
        if self.position is None:
            self.get_init_response()

    def position_to_text(self):
        return str(self.position)
    
    def parse_solution(self, candidate_output):
        return candidate_output
    
    def get_feedback(self):
        return self.feedback.get("feedback", "No feedback")
    
    def is_terminal(self):
        if self.feedback is None:
            self.revise_answers()
        if "correct" in self.feedback:
            return self.feedback["correct"]
        elif "score" in self.feedback:
            return self.feedback["score"] == 10
        else:
            return False

    def get_init_response(self):
        prompt = self.initial_prompt.format(instruction=self.instruction)
        output = self.model([prompt])[0]
        self.position = self.parse_solution(output)
        return self.position

    def revise_answers(self):
        prompt = self.revise_prompt.format(instruction=self.instruction, solution=self.position)
        output = self.model([prompt])[0]
        feedback = self.parse_json_block(output)
        
        self.feedback = feedback
        if not feedback:
            print(output)
        return self.get_feedback()

    def propose_candidate(self, feedback):
        solution = self.position_to_text()
        prompt = self.propose_prompt.format(solution=solution, feedback=feedback, instruction=self.instruction)
        candidate_output = self.model([prompt])[0]
        position = self.parse_solution(candidate_output)
        return position
    

    def get_candidates(self, num_candidates=3):
        if not self.feedback:
            feedback = self.revise_answers()
        else:
            feedback = self.get_feedback()
        candidates = []
        for i in range(num_candidates):
            candidate_position = self.propose_candidate(feedback)
            if candidate_position in [c.position for c in candidates]:
                continue
            candidate = self.__class__(parent=self, instruction=self.instruction, position=candidate_position,
                                        model=self.model)
            if candidate == self:
                continue
            candidates.append(candidate)
        return candidates
    
    def parse_json_block(self, output):
        pattern = r'.*```json\n(.*)\n```.*'
        matched = re.match(pattern, output, re.DOTALL)
        try: 
            if matched:
                json_block = matched.group(1)
                return json.loads(json_block)
        except:
            pass
        return {}

    def parse_feedback(self, feedback):
        feedback_dict = self.parse_json_block(feedback)
        return feedback_dict.get("feedback", "No feedback")

    def reward(self):
        if not self.feedback:
            self.revise_answers()
        if "score" in self.feedback:
            return self.feedback["score"]
        elif "correct" in self.feedback:
            return 10 if self.feedback["correct"] else 0
        else:
            return 0
    

class Node():
    """A node class for Pathfinding"""

    def __init__(self, parent=None, position=None, **kargs):
        self.parent = parent
        self.position = position
        self.f = None
        self.h = None
        self.g = None
        self.use_feedback = kargs.get("use_feedback", True)
        self.external_feedback = kargs.get("external_feedback", False)

    def is_terminal(self):
        if self.f is None:
            self.compute_scores()
        return self.f > 0.95
    
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
        self.f = (self.g + self.h)/2


class ContentNode(Node):
    def __init__(self, model, 
                 external_model=None, parent=None, position=None, **kwargs):
        super().__init__(parent=parent, position=position, **kwargs)
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

    
    def get_future_score(self):
        return 1
    
    def get_current_score(self): # Use an independent evaluator or GPT 
        if self.external_model is None:
            from thoughtsculpt.model.tasks.outline import EVALUATE_CURRENT
            outlines_text = self.position_to_text()
            prompt = EVALUATE_CURRENT.format(outlines_text)
            output = self.model([prompt])[0]
            score, reason = ContentNode.parse_score_reason(output, prompt)
            self.g = score / 100
        else:
            return self.external_model.predict_interestingness(self.position) / 100
            
    
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
    def __init__(self, model, env, external_model=None, parent=None, position=None, **kwargs):
        super().__init__(parent=parent, position=position, **kwargs)
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

            
    
    def get_future_score(self):
        count = self.env.prompt_status(self.model)
        score = count["impossible"] * 0.1 / 10 + count["maybe"] * 0.5 / 10 + count["sure"] / 10
        return score / 3
    
    def get_current_score(self):
        if self.external_feedback:
            num_filled_words = np.count_nonzero(self.position["status"])
            word_score = num_filled_words/10
            num_filled_letters = 25 - self.__str__().count("_")
            letter_score = num_filled_letters/25
            self.g = (word_score + letter_score)/2
        else:
            self.g = 1
        return self.g
    
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
    def __init__(self, model, concepts, parent=None, position=None, **kwargs):
        super().__init__(parent=parent, position=position, **kwargs)
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
        if self.g is None:
            from thoughtsculpt.model.tasks.commongen import EVALUATE_CURRENT
            prompt = EVALUATE_CURRENT.format(self.instruction, self.position)
            output = self.model([prompt])[0]
            score = CommonNode.parse_score(output)
            self.g = score
        return self.g

    def get_future_score(self):
        if self.external_feedback:
            self.h = len([c for c in self.concepts if c in self.position])/len(self.concepts)
        else:
            self.h = 0
        return self.h
            
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
        from thoughtsculpt.model.tasks.commongen import NEW_CANDIDATE
        candidates = []
        if self.use_feedback:
            feedback = self.get_feedback()
        else:
            feedback = "no feedback"
        
        if feedback == 0 and self.h == 1:
            return candidates
        prompt = NEW_CANDIDATE.format(instruct=self.instruction, solution=self.position, feedback=feedback)
        outputs = self.model([prompt]*num_candidates)
        for output in outputs:
            candidate = CommonNode(model=self.model, concepts=self.concepts, 
                                   parent=self, position=CommonNode.parse_sentence(output),
                                   use_feedback=self.use_feedback)
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
        
        
        
        
        
        
        
    
