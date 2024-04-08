import torch
from transformers import T5Tokenizer, AutoModelForSequenceClassification
import numpy as np
import operator
import re
from thoughtsculpt.chatGPT_API import load_model
from thoughtsculpt.model.utils import get_mask


class Evaluator:
    def __init__(self, model_name):
        self.model_name = model_name

    def evaluate_score(self, *args, **kwargs):
        return 0
    
class Comparer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def compare(self, ansA, ansB):
        choices = [0, 1, 2, 3]
        return np.random.choice(choices)

class ContentComparer(Comparer):
    def __init__(self, model_name="gpt-4-0613"):
        self.model_name = model_name
        self.model = load_model(model_name)
        from thoughtsculpt.model.tasks.outline import COMPARE
        self.compare_prompt= COMPARE

    def compare(self, ansA, ansB):
        prompt = self.compare_prompt.format(ansA, ansB)
        output = self.model([prompt])[0]
        return ContentComparer.parse_output(output)
    
    @staticmethod
    def parse_output(output):
        choices = [
            "the better outline is a",
            "the better outline is b",
            "the outlines are equally good",
            "neither outline is good"
        ]
        output = output.lower()
        for i, choice in enumerate(choices):
            if choice in output:
                return i
        return 3
    
class GPTContentEvaluator(Evaluator):
    def __init__(self, model_name="gpt3.5"):
        super(GPTContentEvaluator, self).__init__(model_name)
        self.model = load_model(model_name)
        from thoughtsculpt.model.tasks.outline import EVALUATE_ITEMS
        self.prompt = EVALUATE_ITEMS
    
    def create_prompt(self, outlines, num_mask):
        outline_text = "\n".join([f"[{i+1}] {outline}"  for i, outline in enumerate(outlines)])
        prompt = self.prompt.format(outline_text, num_mask)
        return prompt
    
    def predict_least_interesting(self, outlines, num_mask=2):
        prompt = self.create_prompt(outlines, num_mask)
        output = self.model([prompt])[0]
        scores = self.get_scores(output)
        least_idx = min(scores.items(), key=operator.itemgetter(1))[0]
        return least_idx[0] - 1, least_idx[1] - 1 
    
    @staticmethod
    def get_scores(output):
        results = GPTContentEvaluator.parse_output(output)
        scores = {}
        for start, end, score in results:
            scores[start, end] = score
        return scores
    
    @staticmethod
    def parse_output(output):
        results = []
        pattern = r'.* \[(\d+)\]-\[(\d+)\] \[interesting level: (\d+)\].*$'
        for line in output.split("\n"):
            match = re.match(pattern, line)
            start, end, score = match.group(1), match.group(2), match.group(3)
            results.append((int(start), int(end), int(score)))
        return results
                
        
        
    def evaluate_score(self, outlines, mask=None):
        pass
        

class T5_Evaluator(Evaluator):
    def __init__(self, model_name, task="outline", ckpt_dir=None):
        super(T5_Evaluator, self).__init__(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # config = transformers.T5Config.from_config(model_name)
        if ckpt_dir is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        if task == "outline":
            from thoughtsculpt.model.tasks.outline import PROMPT, MASKED_PROMPT
            self.prompt = PROMPT
            self.masked_prompt = MASKED_PROMPT
        
    def load(self, ckpt_path):
        self.model = self.model.from_pretrained(ckpt_path)
    
    @staticmethod
    def outlines_to_text(outlines, mask=None):
        if mask is not None:
            return "\n".join([f"[{i+1}] {outline}" if mask[i] else f"[{i+1}*] {outline}" for i, outline in enumerate(outlines)])
        else:
            return "\n".join([f"[{i+1}] {outline}" for i, outline in enumerate(outlines)])
    
    def prepare_input(self, text_input):
        inputs = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs
    
    def create_prompt(self, outlines, mask):
        outline_text = "\n".join([f"[{i+1}] {outline}" if mask[i] else f"[{i+1}*] {outline}" for i, outline in enumerate(outlines)])
        filled_indices = np.where(np.array(mask) == 0)[0]
        start_idx = filled_indices[0] + 1
        end_idx = filled_indices[-1] + 1
        if start_idx == end_idx:
            masked_text = f"[{start_idx}*]"
        else:
            masked_text = ",".join([f"[{i+1}*]" for i in filled_indices])
        prompt = self.masked_prompt.format(outline_text, masked_text)
        return prompt
    
    def predict(self, outlines, mask, print_prompt=False):
        prompt = self.create_prompt(outlines, mask)
        inputs = self.prepare_input(prompt)
        if print_prompt:
            print(prompt)

        outputs = self.model(**inputs)
        pred = outputs.logits.argmax(-1)
        return pred

    def predict_interestingness(self, outlines, mask=None):
        if mask is None:
            mask = get_mask(len(outlines), 0, len(outlines)-1)
        prompt = self.create_prompt(outlines, mask)
        inputs = self.prepare_input(prompt)
        with torch.no_grad():
            outputs = self.model(**inputs)
        score = outputs.logits.softmax(-1)[0][1].item()
        return score
    
    def evaluate_score(self, outlines, mask=None):
        return self.predict_interestingness(outlines, mask)

    def predict_least_interesting_batched(self, outlines, num_mask=2):
        num_items = len(outlines)
        masked_result = []
        masked_indices = []
        prompts = []
        batch_size = 16
       
        with torch.no_grad():
            for i in range(0, num_items):
                for j in range(i+1, min(i+num_mask+1, num_items+1)):
                    mask = get_mask(num_items, i, j)
                    prompt = self.create_prompt(outlines, mask)
                    prompts.append(prompt)
                    masked_indices.append((i, j))
            for i in range(0, len(prompts), batch_size):
                batched_prompts = prompts[i:i+batch_size]
                inputs = self.prepare_input(batched_prompts)
                outputs = self.model(**inputs)
                pred = outputs.logits.softmax(-1)[:,1].tolist()
                masked_result.extend(pred)

        # Getting least interesting
        ranked_indices = sorted(range(len(masked_result)), key=lambda k: masked_result[k])
        indices_list = [masked_indices[i] for i in ranked_indices]
        index = np.argmin(masked_result)
        least_interesting_index = masked_indices[index]
        return least_interesting_index, indices_list
        
class CrosswordEvaluator(Evaluator):
    def __init__(self, env):
        self.env = env

    
    def evaluate_score(self, position):
        env = self.env
        env.reset(**position)
        r_all = (env.board == env.board_gt)
        r_letter = sum(a == b for a, b in zip(env.board, env.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(env.ans, env.ans_gt)) / 10
        return {"r_word":r_word, "r_letter":r_letter}
    






