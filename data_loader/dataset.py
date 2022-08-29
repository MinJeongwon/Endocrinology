from logging import NullHandler
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class Dataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len, test_set = False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test_set = test_set

        df = pd.read_csv(dataset, index_col=0) 
        labelEncoder = LabelEncoder()
        df['label'] = labelEncoder.fit_transform(df['category'])
        self.classes = labelEncoder.classes_
        self.encoded_classes = pd.unique(df['label'])
        self.targets = df['label'].to_numpy()

        self.titles = df['TI'].to_numpy()
        self.abstracts = df['abst'].to_numpy()
        self.df = df

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        title = str(self.titles[index])
        abstract = str(self.abstracts[index])
        target = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            title,
            abstract,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        if self.test_set:
            return {
                'title': title,
                'abstract': abstract,
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
            }
        else:
            return {
                'title': title,
                'abstract': abstract,
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.long)
            }
    
    def get_info(self):
        return self.classes, self.encoded_classes, self.df.shape

    def get_dataframe(self):
        return self.df
