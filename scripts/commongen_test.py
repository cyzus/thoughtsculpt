from thoughtsculpt.chatGPT_API import load_model
from thoughtsculpt.model.improver import CommonGenImprover
from thoughtsculpt.data.commongen_dataset import CommonGenDataset
from thoughtsculpt.options.options import get_eval_args, solver_classes, model_name_dict

from tqdm import tqdm
import numpy as np
import json
import datetime
import os



if __name__ == "__main__":
    
    dataset = CommonGenDataset("../thoughtscult/datasets/commongen/commongen_hard.jsonl")
    args = get_eval_args()
    model_name = model_name_dict[args.model]
    model = load_model(model_name)
    depth = args.depth
    mode = args.mode
    solver = args.solver
    content_improver = CommonGenImprover(model=model, solver_class=solver_classes[solver])

    fdir = f'../thoughtscult/logs/commongen/{model_name}_{solver}_{depth}.json'
    if os.path.exists(fdir):
        with open(fdir, 'r') as f:
            scores = eval(f.read())
    else:
        scores = {"base":[], "improved":[], "origs":[], "news":[]}


    
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    date_time_string = current_datetime.strftime("%m-%d-%H:%M")
    for i in tqdm(range(len(scores["base"]), len(dataset))):
        content_improver.improve(concepts=dataset[i], depth=depth, scores=scores)
        with open(fdir, 'w') as fout:
            json.dump(scores, fout)



