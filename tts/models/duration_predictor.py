import torch.nn as nn
import torch


class DurationPredictor(nn.Module):
    
    def __init__(self, hidden_size: int, kernel_size: int, dropout: float):
        super().__init__()

        self.duration_predictor = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: tensor with shape of (batch_size, seq_len, embed_size)

        Returns: 
            out: tensor with shape of (batch_size, seq_len)
        """

        for idx, operation in enumerate(self.duration_predictor):
            if idx == 0 or idx == 4:
                x = operation(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = operation(x)

        durations = x.squeeze(-1)

        return durations 
