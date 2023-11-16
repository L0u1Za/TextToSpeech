import logging
import torch
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    num_items = len(dataset_items)
    max_audio_length = max([item['audio'].shape[1] for item in dataset_items])
    max_spec_length = max([item['spectrogram'].shape[2] for item in dataset_items])
    max_mel_length = max([item['mel'].shape[2] for item in dataset_items])
    max_text_encoded_length = max([item['text_encoded'].shape[1] for item in dataset_items])
    max_phone_encoded_length = max([item['phone_encoded'].shape[1] for item in dataset_items])
    max_pitch_length = max([item['pitch'].shape[0] for item in dataset_items])
    max_energy_length = max([item['energy'].shape[0] for item in dataset_items])

    audio, spectrogram, mel = torch.zeros(num_items, max_audio_length), torch.zeros(num_items, dataset_items[0]['spectrogram'].shape[1], max_spec_length), torch.zeros(num_items, dataset_items[0]['mel'].shape[1], max_mel_length)
    duration, text, phone, audio_path = [], [], [], []
    text_encoded = torch.zeros(num_items, max_text_encoded_length)
    phone_encoded = torch.zeros(num_items, max_phone_encoded_length)
    phone_duration = torch.zeros(num_items, max_phone_encoded_length)
    pitch, energy = torch.zeros(num_items, max_pitch_length), torch.zeros(num_items, max_energy_length)
    spectrogram_length, text_encoded_length, phone_encoded_length = torch.tensor([item['spectrogram'].shape[2] for item in dataset_items], dtype=torch.int32), \
                                                                    torch.tensor([item['text_encoded'].shape[1] for item in dataset_items], dtype=torch.int32), \
                                                                    torch.tensor([item['phone_encoded'].shape[1] for item in dataset_items], dtype=torch.int32)

    for i, item in enumerate(dataset_items):
        audio[i, :item['audio'].shape[1]] = item['audio'].squeeze(0)
        spectrogram[i, :item['spectrogram'].shape[1], :item['spectrogram'].shape[2]] = item['spectrogram'].squeeze(0)
        mel[i, :item['mel'].shape[1], :item['mel'].shape[2]] = item['mel'].squeeze(0)
        text_encoded[i, :item['text_encoded'].shape[1]] = item['text_encoded'].squeeze(0)
        phone_encoded[i, :item['phone_encoded'].shape[1]] = item['phone_encoded'].squeeze(0)

        if 'phone_durations' in item:
            phone_duration[i, :len(item['phone_durations'])] = torch.tensor(item['phone_durations'])
        if 'pitch' in item:
            pitch[i, :item['pitch'].shape[0]] = item['pitch']
        if 'energy' in item:
            energy[i, :item['energy'].shape[0]] = item['energy']
        text.append(item['text'])
        duration.append(item['duration'])
        audio_path.append(item['audio_path'])
        phone.append(' '.join(item['phones']))

    src_pos = list()
    max_len = int(max(phone_encoded_length))
    for length_src_row in phone_encoded_length:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel1 in mel:
        length_mel = np.append(length_mel, mel1.size(1))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    result_batch = {
        "audio": audio,
        "spectrogram": spectrogram,
        "mel": mel,
        "duration": duration,
        "text": text,
        "text_encoded": text_encoded,
        "phone": phone,
        "phone_encoded": phone_encoded.long(),
        "phone_duration": phone_duration.long(),
        "audio_path": audio_path,
        "spectrogram_length": spectrogram_length,
        "text_encoded_length": text_encoded_length,
        "phone_encoded_length": phone_encoded_length,
        "pitch": pitch,
        "energy": energy,
        "mel_pos": mel_pos.long(),
        "src_pos": src_pos.long(),
        "mel_max_len": max_mel_len
    }
    return result_batch