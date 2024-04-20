from pathlib import Path
from src.utils.vocab import Vocab
from src.data.wmt_zh_en_dataset import WmtZhEnDataset

def test_vocab():
    #data_path = Path(__file__).parent.parent / 'dataset/wmt_zh_en/train_10k.csv'
    data_path = '/Users/alkaid/dev/lab/transformer/src/dataset/wmt_zh_en/raw.csv'
    dataset = WmtZhEnDataset(data_path)



    zh_text_list = []
    en_text_list = []

    for text_pair in dataset:
        zh_text_list.append(text_pair[0])
        en_text_list.append(text_pair[1])


    zh_vocab = Vocab()
    zh_vocab.build(zh_text_list)
    zh_vocab.save(Path(__file__).parent.parent / 'dataset/wmt_zh_en/zh_vocab.pkl')


    en_vocab = Vocab()
    en_vocab.build(en_text_list)
    en_vocab.save(Path(__file__).parent.parent / 'dataset/wmt_zh_en/en_vocab.pkl')

    print(len(zh_vocab))
    print(zh_vocab.word2id)
    print(zh_vocab.id2word)


    print(len(en_vocab))
    print(en_vocab.word2id)
    print(en_vocab.id2word)
