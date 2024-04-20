
import torch

from src.utils import Vocab, get_optimizer, initialize_weights

from src.model import make_model
from src.trainer import TranslateTrainer
from .data.demo_data_loader import DemoDataloader

def test_train():
    V = 11
    src_vocab = Vocab(min_freq=0)
    src_vocab.build(['我 是 兔兔，你是谁？'])
    tgt_vocab = Vocab(min_freq=0)
    tgt_vocab.build(['I am a tiger, you are who?'])

    train_data_loader = DemoDataloader(100, l=10000, V=11, max_seq_length=10)
    validation_data_loader = None#DemoDataloader(100, l=1000, V=11, max_seq_length=10)

    # model 
    model = make_model(V, V, 6)

    # optimizer and loss_fn
    param = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer('adam', param, lr=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    #trainer
    model.apply(initialize_weights)
    trainer = TranslateTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, cfg=dict(), 
                            train_data=train_data_loader, validation_data=validation_data_loader, logger=None,
                            src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    

    trainer.train()
    