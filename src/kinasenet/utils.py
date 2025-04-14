import numpy as np
import pandas as pd
import os
import random
import copy
import warnings
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune


def set_seeds(seed_value=42, cuda_deterministic=False):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def merge_duplicated_rows(df, idsep=';'):
    """
    Accepts a pandas DataFrame and merges duplicated rows and concatenate their indices.
    """
    dupdf = df[df.duplicated(keep=False)]

    if dupdf.shape[0] == 0:
        print('No redundancy among rows found, returning input.')
        
        return df

    dupdfix = dupdf.index.values

    merged = []
    for _, val in dupdf.groupby(dupdf.columns.tolist()):
        inx = val.index.tolist()
        i = idsep.join(inx)
        v = val.max(0)
        v.name = i
        merged.append(v)

    df = df.drop(dupdfix)
    df = pd.concat([df, pd.concat(merged, axis=1).T])
    
    return df

def split_prior(prior, rng=np.random.default_rng(42), fraction_gs=0.1):
    """
    Function that splits a matrix by row subsets while preserving at least 1 nonzero element per columns.
    """
    if (prior.sum(1) == 0).sum() != 0:
        print('Prior have 0s in Kinase, can not split.\nReturning without change')
        return prior, None

    assert fraction_gs > 0, f"fraction_gs must be larger than 0, now {fraction_gs}"

    use_prior = prior.copy()
    use_prior = use_prior.loc[:, use_prior.astype(bool).sum(0) != 0].copy()
    use_gs = use_prior.copy()
    
    _, cols = use_prior.shape

    numzero = 1
    counter = 0
    while numzero > 0:
        colsids = np.arange(cols)

        rng.shuffle(colsids)
        colsgs = sorted(colsids[:int(cols * fraction_gs)])
        colspri = sorted(colsids[int(cols * fraction_gs):])
        
        tmp = use_prior.copy()
        tmp.iloc[:, colsgs] = 0
        numzero = (tmp.astype(bool).sum(1) == 0).sum()

        counter += 1

        if not (counter % 100):
            print(f'Attemts to create full prior: {counter}', end='\r')

    use_prior.iloc[:, colsgs] = 0
    use_prior = use_prior.reindex(columns=prior.columns, fill_value=0)

    use_gs.iloc[:, colspri] = 0
    use_gs = use_gs.loc[:, use_gs.astype(bool).sum(0) != 0].copy()

    if use_prior[use_gs.columns].astype(bool).sum().sum() > 0:
        warnings.warn('Overlap between prior and gold standard exist.')

    if (use_prior.sum(1) == 0).sum():
        warnings.warn('There appear to be kinases in prior that do not have phosphosite connections. Something is wrong!')
    
    use_gs = use_gs[use_gs.sum(1) > 0]

    return use_prior, use_gs

def setup_dataloader_from_df(df, batch_size, shuffle=True):
    if df.shape[0] != 0:
        dataset = TensorDataset(torch.tensor(np.array(df), dtype=torch.float), torch.tensor(np.array(df), dtype=torch.float))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = None

    return data_loader

def cal_cpd(model, data_loader, node_indices, device='cpu'):
    """
    Calculate coefficient of partial determination (CPD) when delete a specific kinase.

    :param model: trained model
    :param data_loader: phos-MS data loader
    :param node_indices: kinase indices to be deleted
    """
    # Save a copy of the original model
    original_model = copy.deepcopy(model)
        
    # Calculate original output
    original_mse = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        model.eval()
        with torch.no_grad():
            original_output = model(inputs)
    
        mse = ((original_output - targets) ** 2).mean(dim=0)
        original_mse += mse
    original_mse = (original_mse / len(data_loader)).detach()

    # Calculate perturbed output
    ## KO kinase
    perturbed_mse = []
    for node_index in node_indices:
        single_perturbed_mse = cal_perturbed_mse(original_model, data_loader, node_index, layer=2, device=device)
        perturbed_mse.append(single_perturbed_mse)
    
    perturbed_mse = torch.stack(perturbed_mse, 0)
    # print(perturbed_mse.size())
    
    kin_cpd = 1 - original_mse / perturbed_mse

    ## KO meta kinase
    perturbed_mse = []
    for node_index in node_indices:
        single_perturbed_mse = cal_perturbed_mse(original_model, data_loader, node_index, layer=3, device=device)
        perturbed_mse.append(single_perturbed_mse)
    
    perturbed_mse = torch.stack(perturbed_mse, 0)    
    mkin_cpd = 1 - original_mse / perturbed_mse

    return kin_cpd, mkin_cpd

def cal_perturbed_mse(original_model, data_loader, node_index, layer=2, device='cpu'):
    model_copy = copy.deepcopy(original_model)
    if layer==2:
        model_copy.set_mask1(node_index)
        print(f"Delete kinase {model_copy.kinase_mask1} in layer 2.", end='\r')
    else:
        model_copy.set_mask2(node_index)
        print(f"Delete meta kinase {model_copy.kinase_mask2} in layer 3.", end='\r')

    perturbed_mse = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        model_copy.eval()
        with torch.no_grad():
            perturbed_output = model_copy(inputs)
    
        mse = ((perturbed_output - targets) ** 2).mean(dim=0)
        perturbed_mse += mse
    perturbed_mse = (perturbed_mse / len(data_loader)).detach()

    return perturbed_mse

def threshold_cpd(cpd, eps):
    cpd[cpd.abs() < eps] = 0
    cpd_mask = (cpd > 0).int()
    cpd = cpd * cpd_mask
    cpd[cpd < eps] = 0

    return cpd, cpd_mask

def cal_kk_mean_cpd(kin_cpd, kin_mask, mkin_cpd, mkin_mask):
    """
    Note that all the input shapes are site * kinase.
    """
    M1 = mkin_cpd * mkin_mask
    M2 = kin_cpd * kin_mask
    
    kk_nnz_joint = mkin_mask.T.float() @ kin_mask.float()
    kk_nnz_joint = torch.nan_to_num(kk_nnz_joint, nan=0.0, neginf=0.0).int()

    kk_mean_cpd = (M1.T @ M2) / kk_nnz_joint
    kk_mean_cpd = torch.nan_to_num(kk_mean_cpd, nan=0.0, neginf=0.0)
    
    return kk_mean_cpd

def get_old_mask(module, masked=True, device='cpu'):
    if hasattr(module.fc, 'weight_mask') and masked:
        old_mask = (module.fc.weight_mask.detach() > 0).to(torch.long)
    else:
        old_mask = torch.ones(module.fc.weight.size(), device=device)

    return old_mask

def prune_step(module, new_mask, fill_zeroed=True):
    if prune.is_pruned(module.fc):
        # Removing previous mask
        prune.remove(module.fc, 'weight')

    indx = (new_mask.sum(1) == 0)
    if fill_zeroed and (indx.sum() > 0):
        new_mask[indx, :] = 1

    prune.custom_from_mask(module.fc, 'weight', new_mask)

def cal_kin_act(data, prior, model, batch_size, meta_kin=False, device='cpu'):
    data_loader = setup_dataloader_from_df(df=data, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    
    kin_act = []
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        
        model = model.eval()
        with torch.no_grad():
            outputs = model.prior(inputs)
            if meta_kin:
                outputs = model.kki(outputs)
                
        kin_act.append(outputs)

    kin_act = torch.vstack(kin_act).detach().cpu().numpy()
    
    if not meta_kin:
        kin_act_df = pd.DataFrame(kin_act, index=data.index, columns=prior.index)
    else:
        kin_act_df = pd.DataFrame(kin_act, index=data.index, columns=[str(col) for col in np.arange(prior.shape[0])])

    return kin_act_df