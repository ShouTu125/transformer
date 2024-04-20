
"""
模型训练
"""
from abc import abstractmethod
import os

import torch
import logging

class Trainer(object):
   def __init__(self, model, optimizer, loss_fn, cfg, train_data, validation_data = None, test_data = None, logger = None):
      self.logger = logger
      if self.logger is None:
         self.logger = logging.getLogger(__name__)
         self.logger.setLevel(logging.DEBUG)
      
      self.model = model
      self.optimizer = optimizer
      self.loss_fn = loss_fn
   

      self.train_data = train_data
      self.validation_data = validation_data
      self.test_data = test_data

      self.start_epoch = 1
      self.cur_epoch = 0
      
      self.cfg = cfg
      self.epochs = cfg.get('epochs', 10)
      self.save_epochs = cfg.get('save_epochs', 1)
      self.early_stop = cfg.get('early_stop', False)
      self.best_val_loss = None
      self.save_dir = cfg.get('save_dir', './')
      self.patience = cfg.get('patience', 10)
      self.early_stop_counter = 0

      if os.path.exists(self.save_dir) is False:
         os.makedirs(self.save_dir)

      if cfg.get('resume', False):
         self._resume_checkpoint(cfg.get('resume_path'))

   @abstractmethod
   def _run_epoch(self, epoch):
      raise NotImplementedError

   def train(self):
      for epoch in range(self.start_epoch, self.epochs + 1):
         self.cur_epoch = epoch
         val_loss, metrics_score = self._run_epoch(epoch)

         if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self._save_checkpoint(epoch, save_best=True)
         elif val_loss > self.best_val_loss:
            if self.early_stop:
               self.early_stop_counter += 1
               self.logger.debug(f'EarlyStopping counter: {self.early_stop_counter} out of {self.patience}')
               if self.early_stop_counter >= self.patience:
                  self.logger.debug('EarlyStopping')
                  break
         else:
            self.best_val_loss = val_loss
            self._save_checkpoint(epoch, save_best=True)
         
         if epoch % self.save_epochs == 0:
            self._save_checkpoint(epoch)
         
   def _save_checkpoint(self, epoch, save_best=False):
      state = {
         'epoch': epoch,
         'state_dict': self.model.state_dict(),
         'optimier': self.optimizer.state_dict(),
         'config': self.cfg,
         'best_val_loss': self.best_val_loss,
      }

      # save checkpoint
      checkpoint_file = ''
      if save_best:
         checkpoint_file = os.path.join(self.save_dir, 'model_best.pth')
      else:
         checkpoint_file = os.path.join(self.save_dir, f'checkpoint_{epoch}.pth')

      torch.save(state, checkpoint_file)
      self.logger.debug(f'Save checkpoint to {checkpoint_file}')

   def _resume_checkpoint(self, checkpoint_path):
      self.logger.debug(f'Loading checkpoint {checkpoint_path}')

      checkpoint = torch.load(checkpoint_path)

      self.start_epoch = checkpoint['epoch'] + 1
      self.best_val_loss = checkpoint['best_val_loss']
      self.cfg = checkpoint['config']
      self.model.load_state_dict(checkpoint['state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimier'])

      self.logger.debug(f'Checkpoint config: {self.cfg}')
      self.logger.debug(f'Resume from epoch {self.start_epoch}')
      self.logger.debug(f'Best val loss: {self.best_val_loss}')

      self.logger.debug(f'Checkpoint {checkpoint_path} loaded.')
