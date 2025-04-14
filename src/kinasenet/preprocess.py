import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

from .utils import merge_duplicated_rows


class RobustMinScaler(RobustScaler):
    """
    Applies the RobustScaler and then adjust minimum value to 0.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        super().fit(X, y=y)

        return self

    def transform(self, X):
        X = super().transform(X)
        data_min = np.nanmin(X, axis=0)
        X -= data_min
        
        return X


class DataProcessor(object):
    """
    Class for data preprocessing.
    
    :param exp_path: path to phos-MS data, site * sample
    :param ksr_path: path to KSRs, site * kinase, with values of 1 or specific scores at known KSRs, and 0 elsewhere
    :param output_path: path to save preprocessed data
    :param with_centering: center the data before scaling if True, default: False
    :param quantile_range: tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, default: (1, 99)
    :param unit_variance: scale data so that normally distributed features have a variance of 1 
                          if True, default: False
    """
    def __init__(self, exp_path=None, ksr_path=None, output_path='./', with_centering=False, quantile_range=(1, 99), unit_variance=False):
        super().__init__()
        
        self.exp_path = exp_path
        self.ksr_path = ksr_path
        self.output_path = output_path
        self.with_centering = with_centering
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance

    def load_data(self):
        # exp
        self.exp = pd.read_feather(self.exp_path)
        self.exp.index = self.exp['index'].to_list()
        self.exp = self.exp.drop(['index'], axis=1)
        self.exp = self.exp.sort_index()

        # ksr
        self.ksr = pd.read_csv(self.ksr_path, sep='\t', index_col=0)
        self.ksr = self.ksr.loc[self.exp.index]

        print(f"Totally {self.exp.shape[0]} phosphosites and {self.exp.shape[1]} samples\n")

    def normalize_data(self):
        transformer = RobustMinScaler(with_centering=self.with_centering, quantile_range=self.quantile_range, unit_variance=self.unit_variance)
        data = transformer.fit_transform(self.exp.T)  ## sample * site
        
        data = pd.DataFrame(data, index=self.exp.columns, columns=self.exp.index)
        self.data = data
                
    def process_ksr(self):
        prior = self.ksr.T.copy()  ## kinase * site
        prior = prior[prior.sum(axis=1)!=0]
        prior = merge_duplicated_rows(prior, idsep=';')
        self.prior = prior
        
        print(f"Total number of merged kinases: {self.prior.shape[0]}\n")
          
    def save_data(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
            
        self.data.to_parquet(os.path.join(self.output_path, 'data.parquet'))
        self.prior.to_parquet(os.path.join(self.output_path, 'prior.parquet'))

        print(f"All preprocessed files are saved to {self.output_path}\n")

    def process_all(self):
        print('Loading data...')
        self.load_data()
            
        print("Executing RobustMinScaler...\n")
        self.normalize_data()
         
        print('Processing KSR...')
        self.process_ksr()

        print('Saving data...')
        self.save_data()

        print('Done!')

        return self.data, self.prior
