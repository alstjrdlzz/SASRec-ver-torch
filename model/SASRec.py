import torch
import torch.nn as nn


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
            self.relu = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.dropout2 = nn.Dropout(dropout)
            
        def forward(self, x):
            x_ = self.dropout1(self.relu(self.conv1(x)))
            x_ = self.dropout2(self.conv2(x))
            return x + x_