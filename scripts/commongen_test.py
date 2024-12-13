from thoughtsculpt.chatGPT_API import load_model
from thoughtsculpt.model.improver import CommonGenImprover
from thoughtsculpt.data.commongen_dataset import CommonGenDataset
from thoughtsculpt.options.options import get_eval_args, solver_classes, model_name_dict

from tqdm import tqdm
import numpy as np
import json
import datetime
import os

cwd = os.getcwd()

def main():
    dataset = CommonGenDataset(f"E:/OneDrive/Documents/Workspace/thoughtsculpt/thoughtsculpt/datasets/commongen/commongen_hard.jsonl")
    args = get_eval_args()
    model_name = model_name_dict[args.model]
    model_name2 = model_name_dict["llama3-1-8b"]
    model = load_model(model_name)
    model2 = load_model(model_name2)
    depth = args.depth
    mode = args.mode
    solver = args.solver
    model2_module = "propose" # "revise" # "propose"
    content_improver = CommonGenImprover(model=model, solver_class=solver_classes[solver],
                                         use_feedback=True, model2=model2, model2_module=model2_module)
    fdir = f'E:/OneDrive/Documents/Workspace/thoughtsculpt/thoughtsculpt/logs/commongen/{model_name}_{solver}_{depth}_{model_name2}.json'
    if os.path.exists(fdir):
        with open(fdir, 'r') as f:
            scores = eval(f.read())
            print(np.mean(scores["improved"]))
            return
    else:
        os.makedirs(os.path.dirname(fdir), exist_ok=True)
        scores = {"base":[], "improved":[], "origs":[], "news":[]}


    
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    date_time_string = current_datetime.strftime("%m-%d-%H:%M")
    for i in tqdm(range(len(scores["base"]), len(dataset))):
        content_improver.improve(concepts=dataset[i], depth=depth, scores=scores)
        
        with open(fdir, 'w') as fout:
            json.dump(scores, fout)
    print(np.mean(scores["improved"]))

if __name__ == "__main__":
    main()


