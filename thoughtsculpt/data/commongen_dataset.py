import pandas as pd
from torch.utils.data import Dataset

class CommonGenDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(path_or_buf=data_path, lines=True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        concepts = row['concepts']
        return concepts
        