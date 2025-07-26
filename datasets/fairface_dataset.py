# datasets/fairface_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class FairFaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image = Image.open(row["file_path"]).convert("RGB")

        gender_label = int(row["gender_label"])
        race_label = int(row["race_label"])
        race_name = row["race"]  # for subgroup evaluation

        if self.transform:
            image = self.transform(image)

        return image, gender_label, race_label, race_name
