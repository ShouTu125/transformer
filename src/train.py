import argparse

import torch
import torch.nn as nn
import numpy as np

from model import make_model
from trainer import TranslateTrainer 
from data import WmtZhEnTextPairDataloader as DataLoader

from config import Config, Logger
from utils import get_optimizer, initialize_weights, Vocab

parser = argparse.ArgumentParser()

# dataset parameter
parser.add_argument('--train_data', type=str, default='dataset/wmt_zh_en/train_10k.csv')
parser.add_argument('--validation_data', type=str, default='dataset/wmt_zh_en/validation.csv')
parser.add_argument('--src_vocab', type=str, default='dataset/wmt_zh_en/zh_vocab.pkl')
parser.add_argument('--tgt_vocab', type=str, default='dataset/wmt_zh_en/en_vocab.pkl')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32)

# model parameter
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--n_heads', type=int, default=8) 
parser.add_argument('--d_ff', type=float, default=2048)
parser.add_argument('--dropout', type=float, default=0.1)

# Loss function and Optimizer parameter
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--l2', type=float, default=0.0)

# train parameter
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--save_epochs', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_path', type=str, default='./saved_models/model_best.pt')
parser.add_argument('--log_step', type=int, default=20)
# config
parser.add_argument('--config_file', type=str, default='./config.json')
#log
parser.add_argument('--log_path', type=str, default='./logs/model.log')
parser.add_argument('--log_level', type=str, choices=['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], default='INFO')
# other
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()

logger = Logger(__name__, level=args.log_level, log_path=args.log_path)
cfg = Config(logger=logger, args=args)
cfg.print_config()
cfg.save_config(cfg.config['config_file'])


torch.manual_seed(cfg.config['seed'])
torch.cuda.manual_seed(cfg.config['seed'])
torch.backends.cudnn.enabled = False
np.random.seed(cfg.config['seed'])

# vocab
src_vocab = Vocab()
src_vocab.load(cfg.config['src_vocab'])
tgt_vocab = Vocab()
tgt_vocab.load(cfg.config['tgt_vocab'])

# data_loader
train_data_loader =  DataLoader(cfg.config['train_data'], cfg.config['batch_size'], cfg.config['shuffle'])
validation_data_loader = None
if cfg.config.get('validation_data', None) is None:
    validation_data_loader = DataLoader(cfg.config['validation_data'], cfg.config['batch_size'], cfg.config['shuffle'])

# model 
device = 'cuda:0' if cfg.config['cuda'] else 'cpu'
model = make_model(len(src_vocab), len(tgt_vocab), cfg.config['n_layers'], 
                   cfg.config['d_model'], cfg.config['d_ff'], cfg.config['n_heads'], cfg.config['dropout'])
model.to(device)
logger.debug(model)

# optimizer and loss_fn
param = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], param, lr=cfg.config['lr'])
loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.word2id['<pad>'])

#trainer
model.apply(initialize_weights)
trainer = TranslateTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, cfg=cfg.config, 
                           train_data=train_data_loader, validation_data=validation_data_loader, logger=Logger)

if __name__ == '__main__':
    trainer.train()
