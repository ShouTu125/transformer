"""
词汇表处理工具
"""

import pickle

from tqdm import tqdm
from collections import Counter

class Vocab(object):
    def __init__(self, min_freq = 10):
        self.word2id = None
        self.id2word = None

        self.vocab = Counter()
        self.min_freq = min_freq
        self.vocab_size = 0


    def __len__(self):
        return self.vocab_size

    def word2id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<unk>']

    def id2word(self, id):
        if id in self.id2word:
            return self.id2word[id]
        else:
            return '<unk>'
        
    def save(self, vocab_path):
        vocab = {
            "vocab" : self.vocab,
            "min_freq" : self.min_freq,
            "word2id" : self.word2id,
            "id2word" : self.id2word,
            "vocab_size" : self.vocab_size
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    def build(self, text_list, token_split = ' '):
        for seq in tqdm(text_list):
            self.vocab.update(seq.split(token_split))
        
        tmp = self.vocab.most_common()
        tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        tokens += [i[0] for i in tmp if i[1] > self.min_freq]

        self.word2id = {word:idx for idx, word in enumerate(tokens)}
        self.id2word = {idx:word for word, idx in self.word2id.items()}

        self.vocab_size = len(self.word2id)

    def load(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.vocab = vocab['vocab']
        self.min_freq = vocab['min_freq']
        self.word2id = vocab['word2id']
        self.id2word = vocab['id2word']
        self.vocab_size = vocab['vocab_size']
        