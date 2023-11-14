from torch import nn
from hw_asr.base import BaseModel

class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, rnn_layers=5, rnn_hidden_size=512, dropout=0.2, **batch):
        super().__init__(n_feats, n_class, **batch)


        channels = [(1, 32), (32, 32)]
        kernels = [(41, 11), (21, 11)]
        strides = [(2, 2), (2, 1)]
        paddings = [(20, 5), (10, 5)]

        self.conv = nn.Sequential(
            nn.Conv2d(*channels[0], kernel_size=kernels[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(channels[0][1]),
            nn.Hardtanh(0, 20),
            nn.Dropout(dropout),
            nn.Conv2d(*channels[1], kernel_size=kernels[1], stride=strides[1], padding=paddings[1]),
            nn.BatchNorm2d(channels[1][1]),
            nn.Hardtanh(0, 20),
            nn.Dropout(dropout)
        )
        conv_output_size = (n_feats + 2 * paddings[0][0] - (kernels[0][0] - 1) - 1) // strides[0][0] + 1
        conv_output_size = (conv_output_size + 2 * paddings[1][0] - (kernels[1][0] - 1) - 1) // strides[1][0] + 1
        conv_output_size = channels[1][0] * conv_output_size
        self.rnn = nn.Sequential(
            nn.GRU(
                input_size=conv_output_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_layers,
                bidirectional=True,
                batch_first=True,
            ),
            #nn.BatchNorm2d(rnn_hidden_size),
            #nn.Dropout(dropout)
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, n_class)

    def forward(self, spectrogram, **batch):
        #print("====start====")
        #print(spectrogram.shape)
        spectrogram = spectrogram.unsqueeze(1) # Batch x Channels(1) x Embed x Time
        #print(spectrogram.shape)

        output = self.conv(spectrogram)
        #print("Conv", output.shape)

        output = output.flatten(1, 2) # Batch x Сhannels * Embed x Time
        #print(output.shape)
        output, _ = self.rnn(output.transpose(1, 2))
        #print("Rnn", output.shape)

        output = self.fc(output) # Batch x n_class
        #print(output.shape)
        #print("====end=====")
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2