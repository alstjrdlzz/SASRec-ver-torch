import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, item_num, hidden_dim, max_len, dropout, device):
        super().__init__()

        self.device = device
        self.item_emb = nn.Embedding(item_num, hidden_dim)
        self.posi_emb = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq, mask):
        batch_size = seq.shape[0]
        seq_len = seq.shape[1]

        position = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        seq = self.dropout(self.item_emb(seq) + self.posi_emb(position))
        
        seq = seq.masked_fill(mask==0, 0)
        return seq
        

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.scale = torch.sqrt(torch.tensor([self.hidden_dim])).to(device)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value) 

        energy = torch.matmul(Q, K.T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        return x, attention
    

class PointwiseFeedforwardNetwork(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.conv2(x))
        return x
        
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, dropout, device):
        super().__init__()
        self.sa_layer_norm = nn.LayerNorm(hidden_dim, 1e-8)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, 1e-8)
        self.self_attention_layer = SelfAttentionLayer(hidden_dim, dropout, device)
        self.pointwise_feedforward_network = PointwiseFeedforwardNetwork(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask):
        seq_ = self.sa_layer_norm(seq)
        seq_, _ = self.self_attention_layer(seq_, seq_, seq_, mask)
        seq = seq + self.dropout(seq_)
        
        seq_ = self.ffn_layer_norm(seq)
        seq_ = self.pointwise_feedforward_network(seq_)
        seq = seq + self.dropout(seq_)
        return seq
    

class PredictionLayer(nn.Module):
    def __init__(self, item_emb, device):
        super().__init__()
        self.item_emb = item_emb
        self.device = device

    def forward(self, seq, pos, neg):
        seq_emb = seq
        pos_emb = self.item_emb(pos).to(self.device)
        neg_emb = self.item_emb(neg).to(self.device)
        pos_logits = torch.sum((pos_emb * seq_emb), dim=-1)
        neg_logits = torch.sum((neg_emb * seq_emb), dim=-1)
        return pos_logits, neg_logits