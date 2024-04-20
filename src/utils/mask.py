import torch

def make_src_mask(src, vocab, device):
    src_mask = (src != vocab.word2id['<pad>']).unsqueeze(1).unsqueeze(2).to(device)

    return src_mask

def make_tgt_mask(tgt, vocab, device):
    tgt_pad_mask =  (tgt != vocab.word2id['<pad>']).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.shape[1]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask

    return tgt_mask
