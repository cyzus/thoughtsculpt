import pandas as pd
from torch.utils.data import Dataset

class Game24Dataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        question = row["Puzzles"]
        return question
