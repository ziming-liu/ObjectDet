import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.parameter import Parameter
import numpy as np

def extract_seq_patches(x,kernel_size,rate):
    b,seq_len,seq_dim = x.shape
    k_size = kernel_size + (rate-1)*(kernel_size-1)
    p_right = (k_size-1) // 2
    p_left = k_size-1 -p_right
    x = torch.cat((torch.zeros(b,p_left,seq_dim).cuda(),x,torch.zeros(b,p_right,seq_dim).cuda()),1)
    xs = [x[:,i:i+seq_len] for i in  range(0,k_size,rate)]
    x = torch.cat(tuple(xs),2)
    return x.reshape((-1,seq_len,kernel_size,seq_dim))

class LocalSelfAttention(nn.Module):
    def __init__(self,heads, d_model,  dv, dk=None,neighbors=1, rate=1,
                 key_size=None, mask_right=False, **kwargs):
        super(LocalSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.dv = dv
        self.out_dim = heads * dv
        self.dk = dk if dk else dv
        self.neighbors = neighbors
        self.rate = rate
        self.mask_right = mask_right
        self.d_model = d_model
        self.attention = MultiHeadAttention(n_head=heads,d_model=d_model,d_k=dk,
                                            d_v=dv)
    def forward(self, x,xs,xs_):
        # extract local feat
        kernel_size = 1+2*self.neighbors
        xp = extract_seq_patches(xs,kernel_size,self.rate)
        # reshape
        b,seq_len,seq_dim = x.shape
        assert seq_dim == self.d_model
        #x = x.reshape(-1,1,seq_dim)

        xp = xp.reshape(-1,kernel_size,seq_dim)
        x = x.reshape(xp.shape[0],-1,seq_dim)
        x = self.attention(x,xp,xp)
        # restore shape
        x = x.reshape(-1, seq_len, self.d_model)
        return x



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output#, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output