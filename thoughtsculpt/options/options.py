import argparse
from thoughtsculpt.model.simulator import MCTS, ToT, SelfRefine, COT, DFS
solver_classes = {
    "mcts" : MCTS,
    "refine" : SelfRefine,
    "realtot": ToT,
    "cot": COT,
    "tot": DFS
}
model_name_dict = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4-0125-preview"
}


def get_eval_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="gpt-3.5")
    parser.add_argument("--solver", type=str, default="mcts")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    return args