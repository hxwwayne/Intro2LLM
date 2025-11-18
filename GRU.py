import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.xh2gate = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.xh2candidate = nn.Linear(input_size + hidden_size, hidden_size)

    
    def forward(self, x_t, h_prev):
        z = torch.cat((x_t, h_prev), dim=-1)
        gates = self.xh2gate(z)
        z_t, r_t = gates.chunk(2, dim=-1)
        z_t = torch.sigmoid(z_t)
        r_t = torch.sigmoid(r_t)

        h_candidate = self.xh2candidate(torch.cat((x_t, r_t * h_prev), dim=-1))
        h_candidate = torch.tanh(h_candidate)

        h_t = (1 - z_t) * h_prev + z_t * h_candidate
        return h_t


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(input_size, hidden_size)
    
    def forward(self, x, h_0=None):

        seq_len, batch_size, _ = x.shape

        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h_0
        
        outputs = []

        for t in range(seq_len):
            x_t = x[t]
            h_t = self.gru_cell(x_t, h_t)
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=0)
        return outputs, h_t
