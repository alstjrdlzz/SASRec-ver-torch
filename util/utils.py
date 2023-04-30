import torch


def make_timeline_mask(seq, seq_pad_idx):
    timeline_mask = (seq != seq_pad_idx).unsqueeze(1).unsqueeze(2)
    return timeline_mask

def make_attention_mask(seq, device):
    '''
    1 0 0 0 0
    1 1 0 0 0
    1 1 1 0 0
    1 1 1 1 0
    1 1 1 1 1
    '''
    time_len = seq.shape[1]
    attention_mask = ~torch.tril(torch.ones((time_len, time_len), dtype=torch.bool, device=device))
    return attention_mask