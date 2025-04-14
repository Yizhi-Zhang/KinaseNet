import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import torch

from .utils import cal_cpd, threshold_cpd


class Evaluator(object):
    def __init__(self, eps=None, output_path='./', device='cpu'):
        super().__init__()
        self.eps = torch.finfo(torch.float).eps if eps is None else eps
        self.output_path = output_path
        self.device = device

    def run_evaluation(self, model, data_loader, prior, gs, cv, weight_decay, refit=None, train_report={}):
        node_indices = list(range(model.hidden_dim))
        kin_cpd, _ = cal_cpd(model, data_loader, node_indices, device=self.device)
        kin_cpd, _ = threshold_cpd(kin_cpd, self.eps)
        
        kin_cpd_df = pd.DataFrame(kin_cpd.cpu().numpy(), index=prior.index, columns=prior.columns)
        w3_df = self.extract_weights(model, prior)
        
        perf_cpd_dict = self.cal_performance(kin_cpd_df.T.abs(), gs.T.abs())
        perf_weight_dict = self.cal_performance(w3_df.T.abs(), gs.T.abs())
        
        perf_cpd_df = self.output_performance(perf_cpd_dict, 'cpd', prior, gs, cv, weight_decay, refit, train_report)
        perf_weight_df = self.output_performance(perf_weight_dict, 'weight', prior, gs, cv, weight_decay, refit, train_report)
        perf_df = pd.concat([perf_cpd_df, perf_weight_df], axis=1).T
        
        if hasattr(self, 'performance'):
            self.performance = pd.concat([self.performance, perf_df], axis=0, ignore_index=True)
        else:
            self.performance = perf_df

    def extract_weights(self, model, prior):
        """
        Specifically extract weights of fc3.
        """
        w3 = getattr(model.ksr.fc, 'weight').data.detach().clone().cpu()
        w3[w3.abs() < self.eps] = 0
        if hasattr(model.ksr.fc, 'weight_mask'):
            mask = getattr(model.ksr.fc, 'weight_mask').data.detach().clone().cpu()
            w3 = w3 * mask
        w3 = pd.DataFrame(w3.T if isinstance(w3, np.ndarray) else w3.T.cpu().numpy(), index=prior.index, columns=prior.columns)

        return w3

    def cal_performance(self, pred, gs):
        """
        Calculate MCC, ACC, F1, and AUPRC based on the provided pred and gs.
        Note that pred and gs are DataFrame with shape of kinase * site.
        """
        pred = pred.reindex_like(gs).copy().fillna(0)
        pred_bool = pred.astype(bool).values.ravel().copy()    
        gs_bool = gs.astype(bool).values.ravel().copy()
        gs_binary = gs_bool.astype(int).copy()
    
        TP = (pred_bool & gs_bool).sum().astype(np.float64)
        TN = (~(pred_bool & gs_bool)).sum().astype(np.float64)
        FP = (pred_bool & ~gs_bool).sum().astype(np.float64)
        FN = (~pred_bool & gs_bool).sum().astype(np.float64)
    
        MCCdn = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        MCCn = TP * TN - FP * FN
    
        if MCCdn == 0:
            print('MCC denominator is 0.')
            MCC = 0
        else:
            MCC = MCCn / np.sqrt(MCCdn)
    
        F1 = 2 * TP / (2 * TP + FP + FN)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        
        AUROC = roc_auc_score(gs_binary, pred.values.ravel())
        precision, recall, _ = precision_recall_curve(gs_binary, pred.values.ravel())
        AUPRC = auc(recall, precision)
        
        return {'MCC': MCC, 'F1': F1, 'ACC': ACC, 
                'AUROC': AUROC, 'AUPRC': AUPRC, 
                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

    def output_performance(self, performance_dict, network_type, prior, gs, cv, weight_decay, refit=None, train_report={}):
        num_train_ksr = prior.astype(bool).sum().sum()
        num_test_ksr = gs.astype(bool).sum().sum()
        
        performance = pd.Series(dtype='object')
        performance['network_type'] = network_type
        performance['cv'] = cv
        performance['weight_decay'] = weight_decay
        performance['refit'] = refit
        performance['num_train_ksr'] = num_train_ksr
        performance['num_test_ksr'] = num_test_ksr
        performance['AUPRC'] = performance_dict['AUPRC']
        performance['AUROC'] = performance_dict['AUROC']
        performance['MCC'] = performance_dict['MCC']
        performance['F1'] = performance_dict['F1']
        performance['ACC'] = performance_dict['ACC']
        performance['TP'] = performance_dict['TP']
        performance['TN'] = performance_dict['TN']
        performance['FP'] = performance_dict['FP']
        performance['FN'] = performance_dict['FN']
        performance['train_loss'] = train_report['train_loss']
        performance['val_loss'] = train_report['val_loss']
        performance['val_r2'] = train_report['val_r2']
        performance['activation'] = train_report['activation']
        performance['drop_p_prior'] = train_report['drop_p_prior']
        performance['drop_p_kki'] = train_report['drop_p_kki']
        performance['drop_p_ksr'] = train_report['drop_p_ksr']

        return performance

    def save_performance(self, save_tmp=True):
        if save_tmp:
            file_path = os.path.join(self.output_path, 'performance_tmp.csv')
        else:
            file_path = os.path.join(self.output_path, 'performance.csv')
        
        self.performance.to_csv(file_path)
        
