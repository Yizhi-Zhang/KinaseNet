import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, num_epochs, output_path,
                 scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                 scheduler_kwargs={'T_max': 10}, device='cpu', alpha=1, threshold_right=2, threshold_left=0):
        super().__init__()
        self.num_epochs = num_epochs
        self.output_path = output_path
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.mse = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.device = device
        self.alpha = alpha
        self.threshold_right = threshold_right
        self.threshold_left = threshold_left

        os.makedirs(os.path.join(self.output_path, 'run_log'), exist_ok=True)

    def run_epochs(self, optimizer, model, train_loader, val_loader, cv, weight_decay, refit=None):
        logfile_path = os.path.join(self.output_path, 'run_log', f"cv{cv}_wd{weight_decay}_refit{refit}")
        os.makedirs(logfile_path, exist_ok=True)
        
        writer = SummaryWriter(logfile_path)
        logfile = open(os.path.join(logfile_path, 'log.txt'), 'a')
        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        
        train_epoch_loss = []
        valid_epoch_loss = []
        valid_epoch_r2 = []
        for epoch in range(self.num_epochs):
            train_loss_avg = self.train_iteration(model, train_loader, optimizer)
            val_loss_avg, val_r2_avg = self.valid_iteration(model, val_loader)

            scheduler.step()

            train_epoch_loss.append(train_loss_avg)
            valid_epoch_loss.append(val_loss_avg)
            valid_epoch_r2.append(val_r2_avg)
        
            # Write to tensorboard
            writer.add_scalar('Train/Loss', train_loss_avg, epoch)
            writer.add_scalar('Val/Loss', val_loss_avg, epoch)
            writer.add_scalar('Val/R2', val_r2_avg, epoch)

            # Write to logfile
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val R2: {val_r2_avg:.4f}", end='\r')
            logfile.write(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val R2: {val_r2_avg:.4f}\n")
        
        print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val R2: {val_r2_avg:.4f}\n")
            
        # Close SummaryWriter
        writer.close()

        # Close logfile
        logfile.flush()
        logfile.close()

        # Save report
        np.save(os.path.join(logfile_path, 'train_epoch_loss.npy'), train_epoch_loss)
        np.save(os.path.join(logfile_path, 'valid_epoch_loss.npy'), valid_epoch_loss)
        np.save(os.path.join(logfile_path, 'valid_epoch_r2.npy'), valid_epoch_r2)

        return {'train_loss': train_epoch_loss[-1], 
                'val_loss': valid_epoch_loss[-1], 
                'val_r2': valid_epoch_r2[-1],  ## the last epoch
                'activation': model.activation.__class__.__name__,
                'drop_p_prior': model.prior.dropout_rate,
                'drop_p_kki': model.kki.dropout_rate,
                'drop_p_ksr': model.ksr.dropout_rate
                }
               
    def train_iteration(self, model, train_loader, optimizer):
        model.train()
        
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            output = model(inputs)
            
            loss = self.criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss_avg = train_loss / len(train_loader)

        return train_loss_avg

    def valid_iteration(self, model, val_loader):
        val_loss, val_r2 = 0, 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            model.eval()
            with torch.no_grad():
                output = model(inputs)
            
            loss_mse = self.mse(output, targets)
            loss = self.criterion(output, targets)

            var = torch.var(targets)
            r2 = 1 - (loss_mse.item() / var)
            
            val_loss += loss.item()
            val_r2 += r2.item()

        val_loss_avg = val_loss / len(val_loader)
        val_r2_avg = val_r2 / len(val_loader)

        return val_loss_avg, val_r2_avg
 