import torch
import torch.nn as nn
from Networks.network_utils import make_activation, MultiheadAttention

class Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(self, args):
        super().__init__()
        
        self.cat_num = args['cat_num']
        self.n_layers = args['Trans_n_layers']
        d_token = args['Trans_d_token']
        n_heads = args['Trans_n_heads'] # d % n_heads == 0 
        d_ffn_factor = 0.3
        self.prenormalization = True
        n_classes = args["n_classes"]
        dr = args["dropout_rate"]
        activation = args["activation"]
        d_hidden = int(d_token * d_ffn_factor)
        
        self.attention = MultiheadAttention(d_token, n_heads, dr)
        
        self.norm0 = nn.LayerNorm(d_token)       
        self.linear0 = nn.Linear(d_token, d_hidden)
        self.linear1 = nn.Linear(d_hidden, d_token)
        self.norm1 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dr)
               
        self.activation = make_activation(activation)
        self.norm = nn.LayerNorm(d_token)
        self.fc = nn.Linear(d_token, n_classes)

    def forward(self, x):
        
        # x에 CLS추가?
        
        for layer_idx in range(self.n_layers):
            is_last_layer = layer_idx + 1 == self.n_layers
            
            x_residual = x.clone()
            if self.prenormalization:
                x_residual = torch.clamp(x_residual, -1e9, 1e9) #  x.max(), x.min()이 매우 크면 layer norm하고 x에 갑자기 nan이 생김! -> 방지를 위해 clip
                x_residual = self.norm0(x_residual) 
            x_residual = self.attention((x_residual[:, :1] if is_last_layer else x_residual), x_residual)  # x[:, :1] -> 3차원 1,1,d_token
            # for the last attention, it is enough to process only [CLS]
           
            x = x + x_residual
            if self.prenormalization:
                x = torch.clamp(x, -1e9, 1e9) #  x.max(), x.min()이 매우 크면 layer norm하고 x에 갑자기 nan이 생김! -> 방지를 위해 clip
                x = self.norm0(x)
                
            x_residual = x.clone()
            if self.prenormalization:
                x_residual = torch.clamp(x_residual, -1e9, 1e9) #  x.max(), x.min()이 매우 크면 layer norm하고 x에 갑자기 nan이 생김! -> 방지를 위해 clip
                x_residual = self.norm1(x_residual)
                
            x_residual = self.linear0(x_residual)
            x_residual = self.activation(x_residual)
            
            x_residual = self.dropout(x_residual)
            x_residual = self.linear1(x_residual)
            
            # x_residual = self.dropout(x_residual)
            x = x + x_residual
            if self.prenormalization:
                x = torch.clamp(x, -1e9, 1e9) #  x.max(), x.min()이 매우 크면 layer norm하고 x에 갑자기 nan이 생김! -> 방지를 위해 clip
                x = self.norm1(x)
                
        x = x[:, 0] # [CLS]            
        x = self.fc(self.activation(self.norm(x)))
        
        x = x.squeeze(-1)
        return x


# %%
if __name__ == "__main__":
    
    data=torch.zeros(1, 50).to('cuda')
    model = Transformer(d_numerical=10, cat_num=40, token_bias=True,
        n_layers= 6, d_token= 100, n_heads= 4,  # d % n_heads == 0
        d_ffn_factor= 0.3, prenormalization=True,
        n_classes=5)
    
    model.cuda()
    # if torch.cuda.device_count() > 1:  # type: ignore[code]
    #     print('Using nn.DataParallel')
    #     model = nn.DataParallel(model)
    
    model(data)
    
    