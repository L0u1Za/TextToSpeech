import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path
import tgt
import numpy as np
import pyworld as pw

import torchaudio
from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    "g2p": "https://drive.google.com/uc?export=download&id=1X8XB-_j6qi9PFQAVwOeomIwLNY8dH6kq"
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, config_parser, data_dir=None, index_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "datasets"
            index_dir.mkdir(exist_ok=True, parents=True)
        else:
            index_dir = Path(index_dir)
        self._index_dir = index_dir

        if config_parser["preprocessing"]["g2p"]:
            print("Loading preprocessed g2p")
            download_file(URL_LINKS['g2p'], dest=self._data_dir / 'g2p.zip')
            g2p_path = self._data_dir / 'TextGrid'
            if not g2p_path.exists():
                shutil.unpack_archive(self._data_dir / 'g2p.zip', self._data_dir)

        index = self._get_or_load_index(part)

        super().__init__(index, config_parser=config_parser, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        train_length = int(0.85 * len(files)) # hand split, test ~ 15%
        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        for i, fpath in enumerate((self._data_dir / "wavs").iterdir()):
            if i < train_length:
                shutil.move(str(fpath), str(self._data_dir / "train" / fpath.name))
            else:
                shutil.move(str(fpath), str(self._data_dir / "test" / fpath.name))
        shutil.rmtree(str(self._data_dir / "wavs"))


    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split('|')[0]
                    w_text = " ".join(line.split('|')[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists(): # elem in another part
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "wav_name": w_id,
                                "path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                            }
                        )
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        text_grid_dir = self._data_dir / 'TextGrid' / 'LJSpeech'
        if text_grid_dir.exists():
            tg_path = text_grid_dir / (data_dict['wav_name'] + '.TextGrid')
            text_grid = tgt.io.read_textgrid(tg_path)
            phones, durations, start, end = self.get_alignment(
                text_grid.get_tier_by_name("phones")
            )
        audio_wave = self.load_audio(audio_path)

        audio_wave = audio_wave[:, int(start * self.config_parser["preprocessing"]["sr"]): int(end * self.config_parser["preprocessing"]["sr"])]
        audio_wave, audio_spec = self.process_wave(audio_wave, name="spectrogram")

        all_dur = sum(durations)
        duration = audio_wave.size(1) / self.config_parser["preprocessing"]["sr"]
        audio_spec = audio_spec[:,:,:all_dur]
        pitch, t = pw.dio(
            audio_wave.numpy()[0].astype('float64'),
            self.config_parser["preprocessing"]["sr"],
            frame_period=self.config_parser["preprocessing"]["stft"]["args"]["hop_length"] / self.config_parser["preprocessing"]["sr"] * 1000,
        )
        pitch = torch.tensor(pw.stonemask(audio_wave.numpy()[0].astype('float64'), pitch, t, self.config_parser["preprocessing"]["sr"]))
        pitch = pitch[:all_dur]

        _, mel_spec = self.process_wave(audio_wave, name="stft")
        mel_spec = mel_spec[:,:,:all_dur]
        energy = torch.linalg.norm(mel_spec, ord=2, dim=1).squeeze(0)
        energy = energy[:all_dur]
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "mel": mel_spec,
            "duration": duration,
            "phones": phones,
            "phone_durations": durations,
            "text": data_dict["text"],
            "text_encoded": self.text_encoder.encode(data_dict["text"]),
            "phone_encoded": self.text_encoder.encode(' '.join(phones)),
            "audio_path": audio_path,
            "pitch": pitch,
            "energy": energy
        }
    def process_wave(self, audio_tensor_wave, name="spectrogram"):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"][name],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec
    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.config_parser["preprocessing"]["sr"] / self.config_parser["preprocessing"]["stft"]["args"]["hop_length"])
                    - np.round(s * self.config_parser["preprocessing"]["sr"] / self.config_parser["preprocessing"]["stft"]["args"]["hop_length"])
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time