import time
import logging
import json
import torch
from transformers import AutoTokenizer
from openai import OpenAI
import os
import argparse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


GPT3_END = 'THE END.'
PRETRAINED_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'text-ada-001', 'text-babbage-001', 'text-curie-001',
                     'text-davinci-001', 'text-davinci-002', 'text-davinci-003']


class GPTWrapper():
    def __init__(self, args, client):
        assert args.gpt3_model is not None
        self.model = args.gpt3_model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = args
        self.summarize = {
            "num_queries": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "maximal_prompt_tokens": 0,
        }
        self.client = client
    
    
    @torch.no_grad()
    def __call__(self, texts, 
                     suffixes=None,
                     max_tokens=None, 
                     top_p=None, 
                     temperature=None,
                     retry_until_success=True, 
                     stop=None, 
                     logit_bias=None, 
                     num_completions=1, 
                     model_string=None): 
        self.summarize['num_queries'] += len(texts)
        if logit_bias is None:
            logit_bias = {}
        if suffixes is not None:
            raise NotImplementedError
        if model_string is None:
            pass
        else:
            model_string = None
        return self._call_helper(texts, max_tokens=max_tokens, top_p=top_p,
                                    temperature=temperature, retry_until_success=retry_until_success, stop=stop,
                                    model_string=model_string)
    
    @torch.no_grad()
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _call_helper(self, texts, max_tokens=None, top_p=None, temperature=None,
                     retry_until_success=True, stop=None, logit_bias=None, num_completions=1, \
                     model_string=None):
        assert model_string in PRETRAINED_MODELS or model_string is None
        if logit_bias is None:
            logit_bias = {}
        outputs = []
        for prompt in texts:
            context_length = len(self.tokenizer.encode(prompt))
            self.summarize['total_prompt_tokens'] += context_length
            self.summarize['maximal_prompt_tokens'] = max(self.summarize['maximal_prompt_tokens'], context_length)
            engine = self.model
            completion = self.client.chat.completions.create(
                model=engine,
                messages = [{'role': 'user', 'content': prompt}],
                temperature=temperature if temperature is not None else self.args.summarizer_temperature,
                stop=stop,
                logit_bias=logit_bias,
                n=num_completions)
            outputs.append(completion.choices[0].message.content)
        self.summarize['total_output_tokens'] += sum([len(self.tokenizer.encode(o)) for o in outputs])
        return outputs
          
    

class ChatGPT3Summarizer():
    def __init__(self, args, client):
        assert args.gpt3_model is not None
        self.model = args.gpt3_model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = args
        self.controller = None
        self.summarize = {
            "num_queries": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "maximal_prompt_tokens": 0,
        }
        self.client = client
        
        

    @torch.no_grad()
    def __call__(self, texts, suffixes=None, max_tokens=None, top_p=None, temperature=None, retry_until_success=True,
                 stop=None, logit_bias=None, num_completions=1, model_string=None):
        assert type(texts) == list
        self.summarize['num_queries'] += len(texts)
        if logit_bias is None:
            logit_bias = {}
        if suffixes is not None:
            raise NotImplementedError
        if model_string is None:
            pass
        else:
            model_string = None
        if self.controller is None:
            return self._call_helper(texts, max_tokens=max_tokens, top_p=top_p,
                                     temperature=temperature, retry_until_success=retry_until_success, stop=stop,
                                     model_string=model_string)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _call_helper(self, texts, max_tokens=None, top_p=None, temperature=None,
                     retry_until_success=True, stop=None, logit_bias=None, num_completions=1, \
                     model_string=None):
        assert model_string in PRETRAINED_MODELS or model_string is None

        if logit_bias is None:
            logit_bias = {}

        outputs = []
        for i in range(len(texts)):
            text = texts[i]
            prompt = text

            retry = True
            num_fails = 0
            while retry:
                try:
                    context_length = len(self.tokenizer.encode(prompt))
                    self.summarize['total_prompt_tokens'] += context_length
                    self.summarize['maximal_prompt_tokens'] = max(self.summarize['maximal_prompt_tokens'], context_length)
                    if context_length > self.args.max_context_length:
                        logging.warning('context length' + ' ' + str(
                            context_length) + ' ' + 'exceeded artificial context length limit' + ' ' + str(
                            self.args.max_context_length))
                        time.sleep(5)  # similar interface to gpt3 query failing and retrying
                        assert False
                    if max_tokens is None:
                        max_tokens = min(self.args.max_tokens, self.args.max_context_length - context_length)
                    engine = self.model if model_string is None else model_string
                    logging.log(21, 'PROMPT')
                    logging.log(21, prompt)
                    logging.log(21, 'MODEL STRING:' + ' ' + self.model if model_string is None else model_string)
                    completion = self.client.chat.completions.create(
                        model=engine,
                        messages=[
                            {'role': 'user', 'content': prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature if temperature is not None else self.args.summarizer_temperature,
                        stop=stop,
                        logit_bias=logit_bias,
                        n=num_completions)
                    gpt3_pair = {'prompt': prompt, 'completion': [completion.choices[j].message.content for j in range(num_completions)]}
                    logfile = open('gpt3_log.txt', 'a')
                    logfile.write(json.dumps(gpt3_pair) + '\n')
                    logfile.close()
                    retry = False
                except Exception as e:
                    logging.warning(str(e))
                    retry = retry_until_success
                    num_fails += 1
                    if num_fails > 20:
                        raise e
                    if retry:
                        logging.warning('retrying...')
                        time.sleep(num_fails)
            outputs += [completion.choices[j].message.content for j in range(num_completions)]
        engine = self.model if model_string is None else model_string
        logging.log(21, 'OUTPUTS')
        logging.log(21, str(outputs))
        logging.log(21, 'GPT3 CALL' + ' ' + engine + ' ' + str(
            len(self.tokenizer.encode(texts[0])) + sum([len(self.tokenizer.encode(o)) for o in outputs])))
        self.summarize['total_output_tokens'] += sum([len(self.tokenizer.encode(o)) for o in outputs])
        return outputs

def load_model(model_name='gpt-3.5-turbo', temp = 0.7):
    import argparse
    # set openai.api_key to environment variable OPENAI_API_KEY
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    args = argparse.Namespace()
    args.gpt3_model = model_name
    args.max_tokens = 2048 # output length
    args.max_context_length = 2048 # input length
    args.summarizer_temperature = temp
    args.summarizer_frequency_penalty = 0.0
    args.summarizer_presence_penalty = 0.0
    gpt3 = GPTWrapper(args, client=client)
    return gpt3

