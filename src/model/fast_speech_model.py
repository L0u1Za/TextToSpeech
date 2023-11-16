import torch
from torch import nn
from src.base import BaseModel
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v)))
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = scaled_dot_product_attention(q, k, v, mask=mask, scale=self.d_k**0.5)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

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
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

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
            encoder_dim // encoder_head,
            encoder_dim // encoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = MasksHandler.get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.pad)
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
            decoder_dim // decoder_head,
            decoder_dim // decoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = MasksHandler.get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad=self.pad)
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
        return self.nn(inputs)

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super().__init__()

    def create_alignment(base_mat, duration_predictor_output):
        N, L = duration_predictor_output.shape
        for i in range(N):
            count = 0
            for j in range(L):
                for k in range(duration_predictor_output[i][j]):
                    base_mat[i][count+k][j] = 1
                count = count + duration_predictor_output[i][j]
        return base_mat

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = self.create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        output, mel_len = self.LR(alpha * x, target, mel_max_length)
        return output, mel_len

class VarianceAdaptor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, dropout):
        super().__init__()

        self.duration_predictor = Predictor(in_channels, filter_channels, kernel_size, dropout)
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = Predictor(in_channels, filter_channels, kernel_size, dropout)
        self.energy_predictor = Predictor(in_channels, filter_channels, kernel_size, dropout)

    def forward(self, inputs):
        durations = self.duration_predictor(inputs)
        outputs = self.LR(inputs, durations)

        pitches = self.pitch_predictor(inputs)
        outputs += pitches

        energies = self.energy_predictor(inputs)
        outputs += energies

        return outputs, (durations, pitches, energies)

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, encoder_dim, encoder_conv1d_filter_size, encoder_head, encoder_n_layer, max_seq_len, pad, encoder_dropout, decoder_dim, decoder_conv1d_filter_size, decoder_head, decoder_n_layer, decoder_dropout, fft_conv1d_kernel, fft_conv1d_padding, variance_filter_size, variance_kernel_size, variance_dropout, **batch):
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

    def forward(self, text, **batch):
        pass