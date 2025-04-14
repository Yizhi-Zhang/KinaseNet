import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .relu0 import ReLU0

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, mask, use_mask_as_weights=False, dropout_rate=0.5, activation=ReLU0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask = mask
        self.use_mask_as_weights = use_mask_as_weights
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        
        self.activation = activation
                
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.fc.weight.data = self.fc.weight.data.mul(self.mask).abs()
        self.fc.weight.data[self.fc.weight.data == -0] = 0
        
        if self.use_mask_as_weights:
            self.fc.weight.data = self.mask
        
        prune.custom_from_mask(self.fc, name='weight', mask=(self.mask > 0).to(torch.long))

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)

        return x

class LatentLayer(nn.Module):
    def __init__(self, input_dim, output_dim, mask=None, dropout_rate=0.01, activation=ReLU0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask = mask
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()
    
        self.activation = activation
                
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=False)
        if self.mask is not None:
            prune.custom_from_mask(self.fc, name='weight', mask=(self.mask > 0).to(torch.long))

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)

        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.01, activation=ReLU0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()
    
        self.activation = activation
                
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.fc.weight.data = self.fc.weight.data.abs()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)

        return x

class KinaseNet(nn.Module):
    """
    Model for reconstructing cancer-specfic KSRs.

    :param input_dim: number of sites
    :param hidden_dim: number of kinases
    :param ksr: sites * kinases
    :param use_ksr_as_weights: whether to use KSR scores as initial weights
    """
    def __init__(self, input_dim, hidden_dim, ksr_mask, ppi_mask=None, use_ksr_as_weights=False, 
                 dropout_rate1=0.5, dropout_rate2=0.01, dropout_rate3=0.01, activation=ReLU0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.use_ksr_as_weights = use_ksr_as_weights
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.activation = activation
        
        self.ksr_mask = None
        self.ppi_mask = None
        
        if not torch.is_tensor(ksr_mask):
            if not isinstance(ksr_mask, np.ndarray):
                self.ksr_mask = torch.tensor(np.array(ksr_mask), dtype=torch.float)
            else:
                self.ksr_mask = torch.tensor(ksr_mask, dtype=torch.float)
        else:
            self.ksr_mask = ksr_mask.to(torch.float)
        
        if ppi_mask is not None:
            if not torch.is_tensor(ppi_mask):
                if not isinstance(ppi_mask, np.ndarray):
                    self.ppi_mask = torch.tensor(np.array(ppi_mask), dtype=torch.float)
                else:
                    self.ppi_mask = torch.tensor(ppi_mask, dtype=torch.float)
            else:
                self.ppi_mask = ppi_mask.to(torch.float)

        assert self.input_dim == self.ksr_mask.size(1),\
        f"Dimension mismatch: ksr has {self.ksr_mask.size(1)} phosphosites, but input_dim is {self.input_dim}."
        
        assert self.hidden_dim == self.ksr_mask.size(0),\
        f"Dimension mismatch: ksr has {self.ksr_mask.size(0)} kinases, but hidden_dim is {self.hidden_dim}."

        self.prior = Encoder(input_dim=self.input_dim, output_dim=self.hidden_dim, mask=self.ksr_mask, 
                             use_mask_as_weights=self.use_ksr_as_weights, dropout_rate=self.dropout_rate1, activation=self.activation)
        self.kki = LatentLayer(input_dim=self.hidden_dim, output_dim=self.hidden_dim, mask=self.ppi_mask, dropout_rate=self.dropout_rate2, activation=self.activation)
        self.ksr = Decoder(input_dim=self.hidden_dim, output_dim=self.output_dim, dropout_rate=self.dropout_rate3, activation=self.activation)
        
        # Initialize the mask for each kinase layer to None
        self.kinase_mask1 = None
        self.kinase_mask2 = None

    def forward(self, x):
        x = self.prior(x)
        if self.kinase_mask1 is not None:
            x[:, self.kinase_mask1] = 0

        x = self.kki(x)
        if self.kinase_mask2 is not None:
            x[:, self.kinase_mask2] = 0
            
        x = self.ksr(x)
        
        return x
    
    def set_mask1(self, mask_index):
        self.kinase_mask1 = mask_index
    
    def set_mask2(self, mask_index):
        self.kinase_mask2 = mask_index