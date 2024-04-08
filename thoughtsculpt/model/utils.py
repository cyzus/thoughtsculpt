import numpy as np
import re
from thoughtsculpt.model.tasks.outline import INTERESTING, NON_INTERESTING, MASKED_OUTLINE_PROMPT
from thoughtsculpt.chatGPT_API import load_model

def parse_numbered_list(text):
    outlines = re.findall(r'\n?\d+\. (.*)', text)
    return outlines

def get_mask(length, start_idx, end_idx):
    mask = np.ones(length)
    mask[start_idx:end_idx] = 0
    return mask

class OutlineSampler:
    def __init__(self, outline, model=None, premise=None):
        if isinstance(outline, list):
            self.outlines = outline
            self.premise = premise
        else:
            self.outlines = outline.get_dict_at(2)
            self.premise = outline.premise
        if model is None:
            self.model = load_model()
        else:
            self.model = model

    def update_outlines(self, new_outlines):
        self.outlines = new_outlines

    
    def sample_mask(self, min_num_mask=1, start=None, num_mask=None):
        num_outlines = len(self.outlines)
        if start is not None:
            start_idx = start
        else:
            start_idx = np.random.randint(0, num_outlines - min_num_mask)

        if num_mask is not None:
            end_idx = start_idx + num_mask
        else:
            end_idx = np.random.randint(start_idx+1, num_outlines)
        
        
        return get_mask(len(self.outlines), start_idx, end_idx)
    

    def create_prompt(self, mask, interested, outlines=None):
        if outlines is None:
            outlines = self.outlines
        outline_text = "\n".join([f"{i+1}. {outline}" if masked else "[INSERT]" for i, (outline, masked) in enumerate(zip(outlines, mask))])

        role = INTERESTING if interested else NON_INTERESTING

        prompt = MASKED_OUTLINE_PROMPT.format(role, outline_text)

        return prompt

    def generate_new_outlines(self, prompt):
        outputs = self.model([prompt])[0]
        outputs = parse_numbered_list(outputs)
        return outputs

    @staticmethod
    def get_mask(length, start_idx, end_idx):
        mask = np.ones(length)
        mask[start_idx:end_idx] = 0
        return mask
    
    @staticmethod
    def parse_numbered_list(text):
        outlines = re.findall(r'\n?\d+\. (.*)', text)
        return outlines
