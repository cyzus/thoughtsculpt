from thoughtsculpt.chatGPT_API import load_model
from thoughtsculpt.model.tasks.crosswords import MiniCrosswordsEnv
from thoughtsculpt.model.improver import CrosswordImprover
from thoughtsculpt.evaluation.content_evaluator import CrosswordEvaluator
from thoughtsculpt.options.options import get_eval_args, solver_classes, model_name_dict

from tqdm import tqdm
import numpy as np
import json
import datetime
import os


if __name__ == "__main__":
    env = MiniCrosswordsEnv()

    content_evaluator = CrosswordEvaluator(env=env) 
    args = get_eval_args()
    model = args.model  
    model_name = model_name_dict[model]
    model = load_model(model_name)
    solver = args.solver
    depth = 20
    content_improver = CrosswordImprover(model=model, evaluator=content_evaluator, solver_class=solver_classes[solver]) 

    fdir = f'../thoughtscult/logs/crosswords/{model_name}_{solver}_{depth}.json'
    if os.path.exists(fdir):
        with open(fdir, 'r') as f:
            scores = eval(f.read())
    else:
        scores = {"r_word":[], "r_letter":[]}

    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    date_time_string = current_datetime.strftime("%m-%d-%H:%M")
    for i in tqdm(range(len(scores["r_word"])*5, 100, 5)):
        env.reset(i)
        content_improver.improve(env, depth=depth, scores=scores)
        with open(fdir, 'w') as fout:
            json.dump(scores, fout)
        print("Summarizer", model.summarize)



