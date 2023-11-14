from torch import nn
from hw_asr.base import BaseModel
import torch

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()

class SE(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, dim // 8),
            Swish(),
            nn.Linear(dim // 8, dim),
        )

    def forward(self, inputs):
        # inputs: Batch x Channels x Time
        inputs_mean = inputs.sum(dim=2) / inputs.shape[1] # Batch x Channels

        tetta = self.model(inputs_mean)
        tetta = tetta.sigmoid().unsqueeze(2) # Batch x Channels x 1
        tetta = tetta.tile((1, 1, inputs.shape[2])) # Batch x Channels x Time

        return tetta * inputs

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=True):
        super().__init__()

        self.kernel_size = 5 # represented in paper
        self.do_activation = activation

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=(self.kernel_size - 1) // 2,
                bias=True
            ),
            nn.BatchNorm1d(num_features=out_channels)
        )
        if (activation):
            self.activation = Swish()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if (self.do_activation):
            outputs = self.activation(outputs)

        return outputs

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=5, stride=1, residual_block=True):
        super().__init__()

        self.kernel_size = 5 # kernel size is represented in paper
        self.residual_block = residual_block
        if (residual_block):
            self.residual = Conv(in_channels, out_channels, stride, activation=False)

        self.conv_layers = nn.ModuleList()

        for i in range(num_layers):
            if (i == 0):
                self.conv_layers.append(Conv(in_channels, out_channels, stride, activation=True))
            else:
                self.conv_layers.append(Conv(out_channels, out_channels, activation=True))

        self.se = SE(out_channels)
        self.activation = Swish()

    def forward(self, inputs):
        outputs = inputs
        for conv_layer in self.conv_layers:
            outputs = conv_layer(outputs)

        outputs = self.se(outputs)
        if (self.residual_block):
            outputs += self.residual(inputs)

        return outputs

class AudioEncoder(nn.Module):
    def __init__(self, input_size, alpha=1):
        super().__init__()

        self.kernel_size = 5 # represented in paper

        size_1 = int(256 * alpha)
        size_2 = int(512 * alpha)
        size_3 = int(640 * alpha)

        self.conv = nn.Sequential(
            ConvBlock(input_size, size_1, num_layers=1, residual_block=False), #0
            ConvBlock(size_1, size_1, num_layers=5), #1
            #ConvBlock(size_1, size_1, num_layers=5), #2
            #ConvBlock(size_1, size_1, num_layers=5, stride=2), #3
            #ConvBlock(size_1, size_1, num_layers=5), #4
            #ConvBlock(size_1, size_1, num_layers=5), #5
            #ConvBlock(size_1, size_1, num_layers=5), #6
            #ConvBlock(size_1, size_1, num_layers=5, stride=2), #7
            #ConvBlock(size_1, size_1, num_layers=5), #8
            #ConvBlock(size_1, size_1, num_layers=5), #9
            #ConvBlock(size_1, size_1, num_layers=5), #10
            ConvBlock(size_1, size_2, num_layers=5), #11
            ConvBlock(size_2, size_2, num_layers=5), #12
            #ConvBlock(size_2, size_2, num_layers=5), #13
            #ConvBlock(size_2, size_2, num_layers=5, stride=2), #14
            #ConvBlock(size_2, size_2, num_layers=5), #15
            #ConvBlock(size_2, size_2, num_layers=5), #16
            #ConvBlock(size_2, size_2, num_layers=5), #17
            #ConvBlock(size_2, size_2, num_layers=5), #18
            #ConvBlock(size_2, size_2, num_layers=5), #19
            #ConvBlock(size_2, size_2, num_layers=5), #20
            #ConvBlock(size_2, size_2, num_layers=5), #21
            ConvBlock(size_2, size_3, num_layers=1, residual_block=False), #21
        )
        # Batch x InputLength x 160

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class LabelEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers = 1, alpha = 1):
        super().__init__()

        size_1 = int(256 * alpha)
        size_2 = int(512 * alpha)
        size_3 = int(640 * alpha)

        self.embedding = nn.Embedding(vocab_size, size_2)
        self.rnn = nn.LSTM(size_2, size_2, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(size_2, size_3)

    def forward(self, inputs):
        embeds = self.embedding(inputs)

        rnn_output, _ = self.rnn(embeds)

        output = self.fc(rnn_output)
        return output

class JointNetwork(nn.Module):
    def __init__(self, vocab_size, output_size):
        super().__init__()
        self.fc = nn.Linear(output_size, vocab_size)

    def forward(self, encoder_outputs, decoder_outputs):
        seq_lengths = encoder_outputs.size(1)
        target_lengths = decoder_outputs.size(1)

        encoder_outputs = encoder_outputs.unsqueeze(2)
        decoder_outputs = decoder_outputs.unsqueeze(1)

        encoder_outputs = encoder_outputs.tile((1, 1, target_lengths, 1))
        decoder_outputs = decoder_outputs.tile((1, seq_lengths, 1, 1))

        output = torch.cat((encoder_outputs, decoder_outputs), dim=-1)

        output = self.fc(output)

        output = output.log_softmax(dim=-1)

        return output

class ContextNetModel(BaseModel):
    def __init__(self, n_feats, n_class, num_layers = 3, alpha = 1, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.encoder = AudioEncoder(n_feats, alpha=alpha)

        self.decoder = LabelEncoder(n_class, num_layers=num_layers, alpha=alpha)

        self.joint = JointNetwork(n_class, int(640 * 2 * alpha))

    def forward(self, spectrogram, text_encoded, **batch):
        encoder_output = self.encoder(spectrogram)

        text_encoded = text_encoded.long()
        decoder_output = self.decoder(text_encoded)

        output = self.joint(encoder_output.transpose(1, 2), decoder_output)

        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths