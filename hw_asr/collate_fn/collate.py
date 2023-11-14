import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    num_items = len(dataset_items)
    max_audio_length = max([item['audio'].shape[1] for item in dataset_items])
    max_spec_length = max([item['spectrogram'].shape[2] for item in dataset_items])
    max_text_encoded_length = max([item['text_encoded'].shape[1] for item in dataset_items])

    audio, spectrogram = torch.zeros(num_items, max_audio_length), torch.zeros(num_items, dataset_items[0]['spectrogram'].shape[1], max_spec_length)
    duration, text, audio_path = [], [], []
    text_encoded = torch.zeros(num_items, max_text_encoded_length)

    spectrogram_length, text_encoded_length = torch.tensor([item['spectrogram'].shape[2] for item in dataset_items], dtype=torch.int32), torch.tensor([item['text_encoded'].shape[1] for item in dataset_items], dtype=torch.int32)

    for i, item in enumerate(dataset_items):
        audio[i, :item['audio'].shape[1]] = item['audio'].squeeze(0)
        spectrogram[i, :, :item['spectrogram'].shape[2]] = item['spectrogram'].squeeze(0)
        text_encoded[i, :item['text_encoded'].shape[1]] = item['text_encoded'].squeeze(0)

        text.append(item['text'])
        duration.append(item['duration'])
        audio_path.append(item['audio_path'])

    result_batch = {
        "audio": audio,
        "spectrogram": spectrogram,
        "duration": duration,
        "text": text,
        "text_encoded": text_encoded,
        "audio_path": audio_path,
        "spectrogram_length": spectrogram_length,
        "text_encoded_length": text_encoded_length
    }
    return result_batch