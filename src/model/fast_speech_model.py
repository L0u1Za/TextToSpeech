import torch
from torch import nn
from src.base import BaseModel
import torch.nn.functional as F
import numpy as np
from src.utils.util import pad


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class MasksHandler:
    @staticmethod
    def get_non_pad_mask(seq, pad):
        assert seq.dim() == 2
        return seq.ne(pad).type(torch.float).unsqueeze(-1)

    @staticmethod
    def get_attn_key_pad_mask(seq_k, seq_q, pad):
        ''' For masking out the padding part of key sequence. '''
        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(pad)
        padding_mask = padding_mask.unsqueeze(
            1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask

    @staticmethod
    def get_mask_from_lengths(lengths, max_len=None):
        if max_len == None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len, 1, device=lengths.device)
        mask = (ids < lengths.unsqueeze(1)).bool()

        return mask


class Encoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, encoder_conv1d_filter_size, encoder_head, max_seq_len, n_layers, pad, dropout, fft_conv1d_kernel, fft_conv1d_padding):
        super(Encoder, self).__init__()

        self.pad = pad
        n_position = max_seq_len + 1
        self.n_head = encoder_head

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=pad
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=pad
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            encoder_dim,
            encoder_dim,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = MasksHandler.get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.pad)
        slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)
        non_pad_mask = MasksHandler.get_non_pad_mask(src_seq, pad=self.pad)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, decoder_dim, decoder_conv1d_filter_size, decoder_head, max_seq_len, n_layers, pad, dropout, fft_conv1d_kernel, fft_conv1d_padding):

        super(Decoder, self).__init__()

        self.n_head = decoder_head
        n_position = max_seq_len + 1
        self.pad = pad

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=pad,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            decoder_dim,
            decoder_dim,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = MasksHandler.get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad=self.pad)
        slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)
        non_pad_mask = MasksHandler.get_non_pad_mask(enc_pos, pad=self.pad)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output

class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)

class Predictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, dropout):
        super().__init__()

        self.nn = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(in_channels, filter_channels, kernel_size, padding=(kernel_size - 1)//2),
            Transpose(1, 2),
            nn.ReLU(),
            nn.LayerNorm(filter_channels),
            nn.Dropout(dropout),
            Transpose(1, 2),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=(kernel_size - 1)//2),
            Transpose(1, 2),
            nn.ReLU(),
            nn.LayerNorm(filter_channels),
            nn.Dropout(dropout),
            nn.Linear(filter_channels, 1)
        )
    def forward(self, inputs):
        return self.nn(inputs).squeeze(-1)

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super().__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class VarianceAdaptor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_bins, encoder_dim, dropout):
        super().__init__()

        pitch_min, pitch_max = (-3, 200)
        self.pitch_quantization = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins - 1), #torch.logspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1),
            requires_grad=False,
        )
        energy_min, energy_max = (-2, 200)
        self.energy_quantization = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1),
            requires_grad=False,
        )

        self.duration_predictor = Predictor(in_channels, filter_channels, kernel_size, dropout)
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = Predictor(in_channels, filter_channels, kernel_size, dropout)
        self.pitch_embedding = nn.Embedding(n_bins, encoder_dim)

        self.energy_predictor = Predictor(in_channels, filter_channels, kernel_size, dropout)
        self.energy_embedding = nn.Embedding(n_bins, encoder_dim)

    def forward(self, inputs, true_duration=None, true_pitch=None, true_energy=None, mel_max_len=None):
        durations = self.duration_predictor(inputs)
        if (self.train and true_duration is not None):
            outputs, _ = self.length_regulator(inputs, duration=true_duration, max_len=mel_max_len)
        else:
            outputs, _ = self.length_regulator(inputs, duration=durations, max_len=mel_max_len)

        pitches = self.pitch_predictor(outputs)
        if (self.train and true_pitch is not None):
            indices = torch.bucketize(true_pitch, self.pitch_quantization, out_int32=True)
        else:
            indices = torch.bucketize(pitches, self.pitch_quantization, out_int32=True)
        pitch_embeds = self.pitch_embedding(indices)
        outputs = outputs + pitch_embeds

        energies = self.energy_predictor(outputs)
        if (self.train and true_energy is not None):
            indices = torch.bucketize(true_energy, self.energy_quantization, out_int32=True)
        else:
            indices = torch.bucketize(energies, self.energy_quantization, out_int32=True)
        energy_embeds = self.energy_embedding(indices)
        outputs = outputs + energy_embeds

        return outputs, (durations, pitches, energies)

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, encoder_dim, encoder_conv1d_filter_size, encoder_head, encoder_n_layer, max_seq_len, pad, encoder_dropout, decoder_dim, decoder_conv1d_filter_size, decoder_head, decoder_n_layer, decoder_dropout, fft_conv1d_kernel, fft_conv1d_padding, variance_filter_size, variance_kernel_size, variance_n_bins, variance_dropout, num_mels, **batch):
        super().__init__()
        self.encoder = Encoder(vocab_size,
                               encoder_dim,
                               encoder_conv1d_filter_size,
                               encoder_head,
                               max_seq_len,
                               encoder_n_layer,
                               pad,
                               encoder_dropout,
                               fft_conv1d_kernel,
                               fft_conv1d_padding)
        self.variance_adaptor = VarianceAdaptor(in_channels=encoder_dim,
                                                filter_channels=variance_filter_size,
                                                kernel_size=variance_kernel_size,
                                                n_bins=variance_n_bins,
                                                encoder_dim=encoder_dim,
                                                dropout=variance_dropout)
        self.decoder = Decoder(decoder_dim,
                               decoder_conv1d_filter_size,
                               decoder_head,
                               max_seq_len,
                               decoder_n_layer,
                               pad,
                               decoder_dropout,
                               fft_conv1d_kernel,
                               fft_conv1d_padding)
        self.head = nn.Linear(decoder_dim,
                             num_mels)

    def forward(self, phone_encoded, src_pos, mel_pos=None, mel_max_len=None, phone_duration=None, pitch=None, energy=None, **batch):
        """ phone_duration, pitch, energy -> train """
        outputs, non_pad_mask = self.encoder(phone_encoded, src_pos)
        outputs, (duration, pitch, energy) = self.variance_adaptor(outputs, phone_duration, pitch, energy, mel_max_len)
        outputs = self.decoder(outputs, mel_pos)
        outputs = self.head(outputs).transpose(1, 2)

        return outputs, (duration, pitch, energy)