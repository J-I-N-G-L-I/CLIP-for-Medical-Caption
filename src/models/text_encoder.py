import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length=5000):
        super().__init__()
        pe = torch.zeros(max_length, embed_dim) # position encoding

        # position.shape = [max_length, 1]
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) # unsqueeze(1) for later broadcasting

        # frequency term
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term) # for even dimension
        pe[:, 1::2] = torch.cos(position * div_term) # for odd dimension

        pe = pe.unsqueeze(0).transpose(0, 1) # pe.shape = [max_len, 1, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=512,
                 max_length=256,
                 num_heads=8,
                 dropout=0.1,
                 num_layers=6,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(embed_dim)

        # TODO: init weights
        # DONE: see _init_weights
        self._init_weights()

    def create_padding_mask(self, x, pad_token: int=0):
        """
        :param x: token sequence [B, sequence_len]
        :param pad_token: ID of padding token

        :return: padding mask [B, sequence_len]
        """
        return x == pad_token

    def forward(self, x, attention_mask=None):
        """

        :param x: x.shape = [B, sequence_len]
        :param attention_mask:
        :return:
        """
        pad_token = 0

        # token embedding
        # token_mask = (x != pad_token) # [B, sequence_len]
        x = self.token_embedding(x) # [B, sequence_len, embed_dim]

        # add pe, but be careful of dimensions
        x.transpose(0, 1) # [sequence_len, B, embed_dim]
        x = self.positional_encoding(x)
        x.transpose(0, 1) # back to [B, sequence_len, embed_dim]

        # get attention mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(x.sum(-1)) # get mask -> [B, sequence_len]

        # encoding from transformer
        encoded = self.encoder(x, src_key_padding_mask=attention_mask) # [B, sequence_len, embed_dim]

        # # pooling
        batch_size = x.shape[0]
        if attention_mask is not None:
            # find the first non-padding feature
            # valid_length = (~attention_mask).sum(dim=1) # [B]
            # output = encoded[torch.arange(batch_size), 0] # for simple implement, just use the first token

            # TODO: Change to average pooling
            # Done
            # average pooling (sum the sequence_len dimension, the whole sentence will be compressed to one vector,
            # so within batch will be [B, embed_dim] -> sequence pooling)
            mask = (~attention_mask).unsqueeze(-1).type_as(encoded) # [B, sequence_len, 1]

            summed = (encoded * mask).sum(dim=1) # [B, embed_dim]
            counts = mask.sum(dim=1).clamp(min=1e-6) # use clamp to avoid dividing 0
            output = summed / counts
        else:
            output = encoded[:, 0]

        output = self.ln(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # since it is the encoder in transformer,
                                                       # this is the best init way (not kaiming init)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)


if __name__ == '__main__':
    vocab_size = 100
    # dummy_input = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)
    dummy_input = torch.randint(low=1, high=vocab_size, size=(10, 100), dtype=torch.long)
    text_encoder = TextEncoder(vocab_size=100)

    with torch.no_grad():
        output = text_encoder(dummy_input)
    print(output.shape) # should be [10, 512]
    print(output)
    print("mean abs:", output.abs().mean())