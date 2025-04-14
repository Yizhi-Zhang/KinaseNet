import numpy as np
import torch

from .utils import set_seeds


class DataSpliter(object):
    """
    Class for spliting phos-MS data.

    :param data: phos-MS data, dataframe, sample * site
    """
    def __init__(self, data, data_val_size=0.5, data_test_size=0):
        super().__init__()

        self.data = data
        self.data_val_size = data_val_size
        self.data_test_size = data_test_size

    def split(self, random_state=42):
        total_samples = self.data.shape[0]
        num_val_samples = int(self.data_val_size * total_samples)
        num_test_samples = int(self.data_test_size * total_samples)
        num_train_samples = total_samples - num_val_samples - num_test_samples
        
        # Randomly shuffle the data
        set_seeds(random_state)
        shuffled_indices = torch.randperm(total_samples).tolist()
        train_indices = shuffled_indices[:num_train_samples]
        val_indices = shuffled_indices[num_train_samples : num_train_samples + num_val_samples]
        test_indices = shuffled_indices[num_train_samples + num_val_samples:]
        
        # split phos-MS data
        self.train_data = self.data.iloc[train_indices,:]
        self.val_data = self.data.iloc[val_indices,:]
        self.test_data = self.data.iloc[test_indices]

        # print(f"train_data size: {self.train_data.shape[0]}, val_data size: {self.val_data.shape[0]}, test_data size: {self.test_data.shape[0]}\n")

        return self.train_data, self.val_data, self.test_data
