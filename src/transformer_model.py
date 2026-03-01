import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
    

class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size=1,
        d_model=128,
        n_heads=8,
        num_layers=3,
        dim_feedforward=256,
        output_horizon=7,
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc_out = nn.Linear(d_model, output_horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        x = self.input_projection(x)
        x = self.pos_encoder(x)

        x = self.transformer(x)

        # Use last timestep representation
        x = x[:, -1, :]

        out = self.fc_out(x)

        return out