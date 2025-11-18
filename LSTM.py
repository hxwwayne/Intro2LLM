import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.xh2gate = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x_t, h_prev, c_prev):
        z = torch.cat((x_t, h_prev), dim=-1)

        gates = self.xh2gate(z)
        f_t, i_t, o_t, g_t = gates.chunk(4, dim=-1)

        f_t = torch.sigmoid(f_t)
        i_t = torch.sigmoid(i_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t) # New candidate cell state

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
    
    def forward(self, x, h_0=None, c_0=None):
        seq_len, batch_size, _ = x.shape
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h_0

        if c_0 is None:
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            c_t = c_0
        
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=0)
        return outputs, (h_t, c_t)
