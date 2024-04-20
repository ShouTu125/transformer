import torch
from torch.nn import functional as F

from .trainer import Trainer

from utils.mask import make_src_mask, make_tgt_mask

class TranslateTrainer(Trainer):
    def __init__(self, model, optimizer, loss_fn, cfg, train_data, validation_data = None, test_data = None, 
                lr_scheduler = None, logger = None, src_vocab=None, tgt_vocab=None):
        super().__init__(model, optimizer, loss_fn, cfg, train_data, validation_data, test_data, logger)

        self.do_validation = True if validation_data is not None else False
        self.do_test = True if test_data is not None else False

        self.device = cfg.get('device', None)
        if self.device is None:
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
            if torch.backends.mps.is_available():
                self.device = 'mps'
        
        logger.info(f'Device is {self.device}')
        
        model.to(self.device)

        self.lr_scheduler = lr_scheduler
        self.log_step = cfg.get('log_step', 10)


        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def _run_epoch(self, epoch):
        self.model.train()

        total_loss = 0
 
        for idx, batch in enumerate(self.train_data):
            self.optimizer.zero_grad()

            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)

            output = self.model(src, tgt, None, None)
            output = output[1:].view(-1, output.shape[-1])

            tgt = tgt[1:].view(-1)
            loss = self.loss_fn(output, tgt)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.log_step > 0 and idx % self.log_step == 0:
                self.logger.info(f'Train Epoch: {epoch}, {idx}/{len(self.train_data)} ({idx * 100 / len(self.train_data):.0f}), Loss: {loss.item():.6f}')
        
        self.logger.info(f'Train Epoch: {epoch}, total Loss: {total_loss:.6f}, mean Loss: {total_loss / len(self.train_data):.6f}')
                         
        if self.do_validation:
            self.logger.debug("Start validation")
            val_loss = self._valid_epoch(self.src_vocab, self.tgt_vocab)

            self.logger.info(f'Train Epoch: {epoch}, validation loss is : {val_loss:.3f}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            return val_loss, None
        
        return total_loss / len(self.train_data), None
    
    def _valid_epoch(self, src_vocab, tgt_vocab):
        self.model.eval()

        val_loss = 0

        with torch.no_grad():
            for _, (src, tgt) in enumerate(self.validation_data):
                src = src.to(self.device)
                tgt = tgt.to(self.device) 

                src_mask = make_src_mask(src, src_vocab, self.device)
                tgt_mask = make_tgt_mask(tgt[:,:-1], tgt_vocab, self.device)

                output = self.model(src, tgt[:,:-1], src_mask, tgt_mask)
                output = F.log_softmax(output, dim=-1)
                output_dim = output.shape[-1]
               

                output = output.contiguous().view(-1, output_dim)
                tgt = tgt[:,1:].contiguous().view(-1)
                
                val_loss += self.loss_fn(output, tgt)
                
        return val_loss / len(self.validation_data)
    