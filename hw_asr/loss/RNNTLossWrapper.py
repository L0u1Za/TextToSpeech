import torch
from torch import Tensor
from torchaudio.functional import rnnt_loss

class RNNTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, logit_lengths, target_lengths):
        return rnnt_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths
        )

class RNNTLossWrapper(RNNTLoss):
    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
                **batch) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        return super().forward(
            logits=log_probs_t,
            targets=text_encoded,
            logit_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
