import torch

def make_src_mask(src, pad, device='cpu'):
    src_mask = (src == pad).unsqueeze(1).unsqueeze(2).to(device)

    return src_mask

def make_tgt_mask(tgt, pad, device='cpu'):
    tgt_pad_mask =  (tgt != pad).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.shape[1]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask

    tgt_mask.to(device)

    return tgt_mask == False



if __name__ == '__main__':
    src = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
    tgt = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])

    src_mask = make_src_mask(src, 0)
    tgt_mask = make_tgt_mask(tgt, 0)

    print(src_mask)
    print(tgt_mask)