import torch
import math
import torch.nn as nn
from torch.autograd import Function

def make_activation(name):
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    else: # name == 'elu':
        return nn.ELU()


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

class FeatureTokenizer(nn.Module):
    
    def __init__(self, tokenize_num, d_numerical, vocab_sizes, cat_num, d_token, bias=True, cls_token=False):
        super().__init__()
        self.cls_token = cls_token
        self.cat_num = cat_num
        
        if cat_num == 0: # no categorical feature
            d_bias = d_numerical
            self.category_embeddings = None
        else: # yes categorical feature
            d_bias = d_numerical + cat_num # bias 개수
            self.category_embeddings = nn.ModuleList([nn.Embedding(int(vocab_sizes[i])+1, d_token) for i in range(cat_num)])
            for i in range(cat_num):
                nn.init.xavier_uniform_(self.category_embeddings[i].weight, gain=1/math.sqrt(2))
        
        if self.cls_token: # ft_transformer의 CLS token 추가인 경우 # [CLS] 함께 고려
            self.weight = nn.Parameter(torch.Tensor(d_numerical+1, d_token))
            self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None # [CLS]
        elif tokenize_num:
            self.weight = nn.Parameter(torch.Tensor(d_numerical, d_token))
            self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        else: # numcerical feature는 tokenize no
            self.weight = nn.Parameter(torch.ones(d_numerical, d_token))
            self.bias = None
            
            
        nn.init.xavier_uniform_(self.weight,  gain=1/math.sqrt(2))
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias,  gain=1/math.sqrt(2))
        
    def forward(self, x_cat, x_num):
        
        if self.cls_token:
            x = self.w_cls_forward(x_cat, x_num)
        else:
            x = self.wo_cls_forward(x_cat, x_num)
        return x
    
    def w_cls_forward(self, x_cat, x_num):
        x_num = torch.cat([torch.ones(x_cat.shape[0], 1, device=x_cat.device)]  # [CLS]
                          + [x_num], dim=1)
        x = self.weight[None] * x_num[:,:,None] # batch size, d_numerical, d_token
        
        # categorical feature 존재 -> nn.Embedding 적용
        if x_cat is not None:
            embedded_x_cat = [self.category_embeddings[i](x_cat[:, None ,i].long()) for i in range(self.cat_num)]
            embedded_x_cat = torch.cat(embedded_x_cat, dim=1)
            x = torch.cat([x, embedded_x_cat], dim=1)
            ## self.category_embedding는 categorical feature * weight
            
        if self.bias is not None:
            cls_bias=torch.zeros(1, self.bias.shape[1]).to(self.bias.device)
            bias = torch.cat([cls_bias, self.bias]) # [CLS] 추가
            x = x + bias
        return x
    
    def wo_cls_forward(self, x_cat, x_num):
        x = self.weight[None] * x_num[:,:,None] # batch size, d_numerical, d_token
        
        # categorical feature 존재 -> nn.Embedding 적용
        if x_cat is not None:
            embedded_x_cat = [self.category_embeddings[i](x_cat[:, None ,i].long()) for i in range(self.cat_num)]
            embedded_x_cat = torch.cat(embedded_x_cat, dim=1)
            x = torch.cat([x, embedded_x_cat], dim=1)
            ## self.category_embedding는 categorical feature * weight
            
        if self.bias is not None:
            x = x + self.bias
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dr):
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dr)
        self.softmax=nn.Softmax(dim=-1)
        
        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
    
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q) 
        k = self._reshape(k)
        attention = self.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key))
        attention = self.dropout(attention)
        
        x = attention @ self._reshape(v) # 행렬 곱셈
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        
        if self.W_out is not None:
            x = self.W_out(x)
            
        return x