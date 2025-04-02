import torch
import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self, input_dim, d_ffn_factor, output_dim, layers, device):
        super(LSTM, self).__init__()
 
        self.hidden_dim = int(input_dim * d_ffn_factor)
        self.output_dim = output_dim
        self.layers = layers
        self.device = device
        
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=layers,
                            dropout = 0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)  
        self.fc = nn.Linear(self.hidden_dim, output_dim, bias = True) 
    
    def init_hidden(self, batch_size):
        """ Initialize hidden and cell states """
        return (torch.zeros(self.layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.layers, batch_size, self.hidden_dim).to(self.device))


    # 예측을 위한 함수
    def forward(self, x):
        h_t, c_t = self.init_hidden(x.size(0)) # # Reset hidden states for each batch
        x, _ = self.lstm(x, (h_t, c_t))
        x = self.dropout(x[:, -1])
        x = self.fc(x)
        # x = self.fc(x[:, -1])
        return x



## regression은 output_dim=1

def lstm_1layers(args):
    args['lstm_n_layers']=1
    d_ffn_factor = args['lstm_hidden_unit_factor']
    return LSTM(args['feature_num'], d_ffn_factor, 1, args['lstm_n_layers'], args['device'])

def lstm_2layers(args):
    args['lstm_n_layers']=2
    d_ffn_factor = args['lstm_hidden_unit_factor']
    return LSTM(args['feature_num'], d_ffn_factor, 1, args['lstm_n_layers'], args['device'])

def lstm_3layers(args):
    args['lstm_n_layers']=3
    d_ffn_factor = args['lstm_hidden_unit_factor']
    return LSTM(args['feature_num'], d_ffn_factor, 1, args['lstm_n_layers'], args['device'])
