import torch.nn as nn
import torch


class FastSpeechMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self, 
        durations_student: torch.Tensor, 
        durations_teacher: torch.Tensor, 
        mel_spec_pred: torch.Tensor, 
        mel_spec_true: torch.Tensor
    ) -> float:
        durations_loss = self.mse_loss(durations_student, durations_teacher)
        mel_spec_loss = self.mse_loss(mel_spec_pred, mel_spec_true)

        return durations_loss + mel_spec_loss
