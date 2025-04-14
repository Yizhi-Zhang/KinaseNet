import numpy as np
import os
import torch

from .split import DataSpliter
from .trainer import Trainer
from .evaluator import Evaluator
from .kinasenet import KinaseNet
from .setup_optimizer import OptimizerSetter
from .relu0 import ReLU0
from .utils import *


def fit_model(data, prior, ppi_mask=None, output_path='./test_result', 
              data_val_size=0.3, batch_size=64, fraction_gs=0.2, 
              num_epochs=30, cvs=10, num_epochs_refit=30, refit_iters=5, refit_resample=True, 
              weight_decays=(-10, -1, 4), lr=1e-4, 
              scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR, scheduler_kwargs={'T_max': 10}, 
              optimizer_class=torch.optim.Adam, optimizerkw={}, optimizer_paramskw={'prior': {'weight_decay': 1e-10}},
              dropout_rate1=0.5, dropout_rate2=0.01, dropout_rate3=0.01, activation=ReLU0,
              eps=torch.finfo(torch.float).eps, eps_factor=10, fill_zeroed=True, device='cpu', 
              alpha = 1, percentile_right = 50, percentile_left = 0):
    """
    The actually scripts for training.
    """
    # Calculate the threshold
    data_copy = np.array(data)
    data_1D = np.array(data_copy).flatten()
    threshold_right = np.percentile(data_1D, percentile_right)
    threshold_left = np.percentile(data_1D, percentile_left)
    # print(f"percentile = {percentile},threshold = {threshold}")

    # Global container
    data_spliter = DataSpliter(data=data, data_val_size=data_val_size, data_test_size=0)
    trainer = Trainer(num_epochs=num_epochs, output_path=output_path,
                    scheduler_class=scheduler_class,
                    scheduler_kwargs=scheduler_kwargs,
                    device=device, alpha=alpha, threshold_right=threshold_right, threshold_left=threshold_left)
    evaluator = Evaluator(eps=eps, output_path=output_path, device=device)

    # Create output dir
    if not os.path.exists(os.path.join(output_path, 'model')):
        os.makedirs(os.path.join(output_path, 'model'), exist_ok=True)

    for cv in range(cvs):
        # 1-1. Split samples and setup dataloader
        train_data, val_data, _ = data_spliter.split(random_state=cv)
        train_loader = setup_dataloader_from_df(df=train_data, batch_size=batch_size, shuffle=True)
        val_loader = setup_dataloader_from_df(df=val_data, batch_size=batch_size, shuffle=False)
        
        # 1-2. Split prior
        rng = np.random.default_rng(cv)
        train_prior, test_prior = split_prior(prior=prior, rng=rng, fraction_gs=fraction_gs)
        
        num_total_ksr = (prior > 0).sum().sum()
        num_train_ksr = (train_prior > 0).sum().sum()
        num_test_ksr = (test_prior > 0).sum().sum()
        print(f"Total number of KSRs: {num_total_ksr}, number of KSRs used to train: {num_train_ksr}, number of KSRs used to test: {num_test_ksr}\n")

        if isinstance(weight_decays, tuple):
            weight_decays = np.logspace(*weight_decays)
        
        for wd in weight_decays:
            print(f"cv: {cv}, weight_decay: {wd}")

            # 2-1. Model initialization
            set_seeds(cv)
            model = KinaseNet(input_dim=train_data.shape[1], hidden_dim=train_prior.shape[0], 
                            ksr_mask=train_prior, ppi_mask=ppi_mask, use_ksr_as_weights=False,
                            dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, 
                            dropout_rate3=dropout_rate3, activation=activation).to(device)
            
            # 2-2. Optimizer initialization
            opt_setter = OptimizerSetter(optimizer_class=optimizer_class, optimizerkw=optimizerkw, 
                                        optimizer_paramskw=optimizer_paramskw, lr=lr, weight_decay=wd, relmax=None, it=None)
            opt_setter.generate_optimizer(model)
    
            # 3. Fit model
            fit_report = trainer.run_epochs(optimizer=opt_setter.optimizer, model=model, train_loader=train_loader, 
                                            val_loader=val_loader, cv=cv, weight_decay=wd, refit=None)
            
            # 4. Evaluate performance
            evaluator.run_evaluation(model=model, data_loader=val_loader, prior=train_prior, gs=test_prior, cv=cv, weight_decay=wd, refit=None, train_report=fit_report)
            evaluator.save_performance(save_tmp=True)
    
            # 5. Save model
            model_save_path = os.path.join(output_path, 'model', f"cv{cv}_wd{wd}_refitNone.pth")
            torch.save(model.cpu(), model_save_path)
            model = model.to(device)
    
            # 6. Refit
            refit_model(model=model, data_spliter=data_spliter, trainer=trainer, evaluator=evaluator, 
                        train_loader=train_loader, val_loader=val_loader, prior=train_prior, gs=test_prior, 
                        batch_size=batch_size, num_epochs=num_epochs_refit, refit_iters=refit_iters, 
                        resample=refit_resample, eps=eps, eps_factor=eps_factor, fill_zeroed=fill_zeroed, 
                        optimizer_class=optimizer_class, optimizerkw=optimizerkw, optimizer_paramskw=optimizer_paramskw, 
                        lr=lr, wd=wd, relmax=0, cv=cv, output_path=output_path, device=device)
    
    # Store final performance
    evaluator.save_performance(save_tmp=False)

def refit_model(model, data_spliter, trainer, evaluator, train_loader, val_loader, 
                prior, gs, batch_size=64, num_epochs=30, refit_iters=5, resample=True, 
                eps=torch.finfo(torch.float).eps, eps_factor=10, fill_zeroed=True, 
                optimizer_class=torch.optim.Adam, optimizerkw={}, 
                optimizer_paramskw={'prior': {'weight_decay': 1e-10}}, lr=1e-4, wd=0,
                relmax=0, cv=0, output_path='./test_result', device='cpu'):
    """
    Prune weights of kki and ksr based on CPD and refit model.
    """
    init_epochs = trainer.num_epochs
    trainer.num_epochs = num_epochs
    
    for refit in range(refit_iters):
        print(f"cv: {cv}, weight_decay: {wd}, refit: {refit}")

        # 1. Resample
        if resample:
            train_data, val_data, _ = data_spliter.split(random_state=refit + 2024)
            train_loader = setup_dataloader_from_df(df=train_data, batch_size=batch_size, shuffle=True)
            val_loader = setup_dataloader_from_df(df=val_data, batch_size=batch_size, shuffle=False)
    
        # 2. Prune based on CPD
        prune_with_cpd(model=model, data_loader=val_loader, eps=eps, eps_factor=eps_factor, fill_zeroed=fill_zeroed, device=device)
            
        # 3. Optimizer initialization
        opt_setter = OptimizerSetter(optimizer_class=optimizer_class, optimizerkw=optimizerkw, 
                                     optimizer_paramskw=optimizer_paramskw, lr=lr, 
                                     weight_decay=wd, relmax=relmax, it=refit)
        opt_setter.generate_optimizer(model)
        # print("refit", refit, opt_setter.optimizer.param_groups[0]['lr'])
    
        # 3. Refit model
        refit_report = trainer.run_epochs(optimizer=opt_setter.optimizer, model=model, train_loader=train_loader, 
                                          val_loader=val_loader, cv=cv, weight_decay=wd, refit=refit)
        
        # 4. Evaluate performance
        evaluator.run_evaluation(model=model, data_loader=val_loader, prior=prior, gs=gs, cv=cv, weight_decay=wd, refit=refit, train_report=refit_report)
        evaluator.save_performance(save_tmp=True)
    
        # 5. Save model
        model_save_path = os.path.join(output_path, 'model', f"cv{cv}_wd{wd}_refit{refit}.pth")
        torch.save(model.cpu(), model_save_path)
        model = model.to(device)

    trainer.num_epochs = init_epochs

def prune_with_cpd(model, data_loader, eps, eps_factor=10, fill_zeroed=True, device='cpu'):
    node_indices = list(range(model.hidden_dim))
    kin_cpd, mkin_cpd = cal_cpd(model, data_loader, node_indices, device=device)  ## kianse * site

    kin_cpd, kin_mask = threshold_cpd(kin_cpd, eps)
    kin_old_mask = get_old_mask(model.ksr, masked=False, device=device)  ## only for getting a matrix of 1s, site * kinase
    kin_new_mask = torch.logical_and(kin_mask.T, kin_old_mask).int()  ## site * kinase
    
    mkin_cpd, mkin_mask = threshold_cpd(mkin_cpd, eps)
    mkin_old_mask = get_old_mask(model.ksr, masked=True, device=device)  ## site * kinase
    mkin_new_mask = torch.logical_and(mkin_mask.T, mkin_old_mask).int()  ## site * kinase

    kk_cpd = cal_kk_mean_cpd(kin_cpd.T, kin_mask.T, mkin_cpd.T, mkin_mask.T)
    _, kk_mask = threshold_cpd(kk_cpd, eps / eps_factor)
    kk_old_mask = get_old_mask(model.kki, masked=True, device=device)
    kk_new_mask = torch.logical_and(kk_mask, kk_old_mask).int()

    prune_step(model.ksr, mkin_new_mask, fill_zeroed=fill_zeroed)
    prune_step(model.kki, kk_new_mask, fill_zeroed=False)
    
    

            
                
            
     