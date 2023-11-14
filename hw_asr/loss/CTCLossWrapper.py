import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
                **batch) -> Tensor:
        #print(log_probs.shape, log_probs_length)
        log_probs_t = torch.transpose(log_probs, 0, 1)
        #print(log_probs_t.shape)
        return super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
