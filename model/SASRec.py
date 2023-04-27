import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout, device):
        super().__init__()
        self.hidden_dim = hidden_dim # latent vector dimension d
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False) # projection matrix W^Q
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False) # projection matrix W^K
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False) # projection matrix W^V

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

        x = torch.matmul(self.dropout(attention), V) # item embedding after self-attention layer S
        return x, attention