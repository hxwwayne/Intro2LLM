import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        ) # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for batch compatibility

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        '''
        x : [batch_size, seq_len, d_model]
        '''

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for query, key, value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # Output linear layer
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, query, key, value, mask=None):
        '''
        query, key, value : [batch_size, seq_len, d_model]
        mask : [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        '''
        batch_size, tgt_len, _ = query.size()
        batch_size, src_len, _ = key.size()

        # Linear projections, [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        def split_heads(x, seq_len=None):
            # Split the last dimension into (num_heads, d_k)
            x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
            return x.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        Q = split_heads(Q, tgt_len)
        K = split_heads(K, src_len)
        V = split_heads(V, src_len)

        scores = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(self.d_k))  # [batch_size, num_heads, tgt_len, src_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)  # [batch_size, num_heads, tgt_len, src_len]
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [batch_size, num_heads, tgt_len, d_k]
        # Concatenate heads [batch_size, num_heads, tgt_len, d_k] -> [batch_size, tgt_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)  # [batch_size, tgt_len, d_model]

        output = self.w_o(context)  # [batch_size, seq_len, d_model]
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        x : [batch_size, seq_len, d_model]
        '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        '''
        x : [batch_size, seq_len, d_model]
        src_mask : [batch_size, 1, 1, seq_len]
        '''
        # self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # feed-forward + residual + norm
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out) 
        x = self.norm2(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_len=5000,
            dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) 
                                     for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, src, src_mask=None):
        '''
        src : [batch_size, seq_len]
        '''
        # src_mask = self.make_src_mask(src, pad_idx) # [batch_size, 1, 1, seq_len]

        # Embedding + Positional Encoding
        x = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        # x_cls = x.mean(dim=1)  # Global average pooling over sequence length [batch_size, d_model]
        # logits = self.fc_out(x_cls)  # [batch_size, num_classes]
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # self-attention
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        # cross-attention query from decoder, key and value from encoder
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        '''
        x : [batch_size, tgt_seq_len, d_model]
        enc_output : [batch_size, src_seq_len, d_model]
        tgt_mask : [batch_size, 1, tgt_seq_len, tgt_seq_len]
        src_mask : [batch_size, 1, 1, src_seq_len]
        '''

        _x = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_len=5000,
            dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) 
                                     for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        '''
        tgt : [batch_size, tgt_seq_len]
        enc_output : [batch_size, src_seq_len, d_model]
        '''

        # Embedding + Positional Encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, tgt_seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return x


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_len=5000,
            dropout=0.1,
            pad_idx=0
    ):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx
    
    def make_src_mask(self, src):
        '''
        src : [batch_size, seq_len]
        returns src_mask : [batch_size, 1, 1, seq_len]
        '''
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask  # 1 for non-pad tokens, 0 for pad tokens
    
    def make_tgt_mask(self, tgt):
        '''
        tgt : [batch_size, tgt_seq_len]
        returns tgt_mask : [batch_size, 1, tgt_seq_len, tgt_seq_len]
        '''
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_seq_len]

        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()  # [tgt_seq_len, tgt_seq_len]
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_seq_len, tgt_seq_len]

        tgt_mask = tgt_pad_mask & subsequent_mask  # Combine masks
        return tgt_mask
    
    def forward(self, src, tgt):
        '''
        src : [batch_size, src_seq_len]
        tgt : [batch_size, tgt_seq_len]
        '''
        src_mask = self.make_src_mask(src)  # [batch_size, 1, 1, src_seq_len]
        tgt_mask = self.make_tgt_mask(tgt)  # [batch_size, 1, tgt_seq_len, tgt_seq_len]

        enc_output = self.encoder(src, src_mask)  # [batch_size, src_seq_len, d_model]
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)  # [batch_size, tgt_seq_len, d_model]

        logits = self.fc_out(dec_output)  # [batch_size, tgt_seq_len, tgt_vocab_size]
        return logits

    def generate(self, src, bos_idx, eos_idx, max_len=50):
        '''
        src : [batch_size, src_seq_len]
        '''
        device = src.device
        batch_size = src.size(0)
        src_mask = self.make_src_mask(src)
        enc_output = self.encoder(src, src_mask)  # [batch_size, src_seq_len, d_model]

        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)  #[batch_size, 1]

        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(tgt)
            dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)  # [batch_size, tgt_seq_len, d_model]
            logits = self.fc_out(dec_output)  # [batch_size, tgt_seq_len, tgt_vocab_size]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]
            tgt = torch.cat([tgt, next_token], dim=1)  # Append predicted token

            if (next_token == eos_idx).all():
                break
        
        return tgt  # [batch_size, generated_seq_len]
    
        












if __name__ == "__main__":
    # Example usage
    vocab_size = 10000
    seq_len = 20
    batch_size = 4
    num_classes = 3

    model = TransformerEncoder(vocab_size = vocab_size, 
                               d_model=128,
                               num_heads=4,
                               num_layers=2,
                               d_ff=256,
                               max_len=seq_len,
                               dropout=0.1,
                               num_classes=num_classes)
    
    src = torch.randint(0, vocab_size, (batch_size, seq_len))  

    pad_idx = 0 

    logits = model(src, pad_idx=pad_idx)
    print(logits.shape)  # Expected output: [batch_size, num_classes]
