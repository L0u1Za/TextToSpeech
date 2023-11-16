import torch
from torch import nn

class FastSpeechLoss(nn.Module):
    def __init__(self, names=["mel_loss", "duration_loss", "pitch_loss", "energy_loss"]):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.names = names

    def forward(self, pred_mel, spectrogram, predicted, predictor_targets, **batch):
        assert len(predicted) == len(predictor_targets)
        mel_loss = self.l1_loss(pred_mel, spectrogram)

        predictor_losses = [self.mse_loss(predicted[i],
                                               predictor_targets[i].float()) for i in range(len(predicted))]

        return mel_loss, *predictor_losses
