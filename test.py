import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH, get_WaveGlow
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

from src.waveglow.inference import inference

import numpy as np


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def synthesis(model, phonemes, alphas, device='cuda'):
    src_pos = np.array([i+1 for i in range(phonemes.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = phonemes.long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel, predictions = model.forward(sequence, src_pos, alphas=alphas)
    return mel.cpu(), mel


def get_data(text_encoder):
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    tests = [
        'EY1 D IY0 F IH1 B R IH0 L EY2 T ER0 IH1 Z EY1 D IH0 V AY1 S DH AH0 T G IH1 V Z EY1 HH AY1 EH1 N ER0 JH IY0 IH0 L EH1 K T R IH0 K SH AA1 K T UW1 DH IY0 HH AA1 R T AH1 V S AH1 M W AH2 N HH UW1 IH1 Z IH1 N K AA1 R D IY0 AE2 K ER0 EH1 S T',
        'M AE2 S AH0 CH UW1 S AH0 T S IH1 N S T AH0 T UW2 T AH1 V T EH0 K N AA1 L AH0 JH IY0 M EY1 B IY1 B EH1 S T N OW1 N F R ER0 IH1 T S M AE1 TH S AY1 AH0 N S AH0 N D EH1 N JH AH0 N IH1 R IH0 NG EH2 JH Y UW0 K EY1 SH AH0 N',
        'W AA1 S ER0 S T IY2 N D IH1 S T AH0 N S ER0 K AE1 N T ER0 OW0 V IH2 CH R UW1 B IH0 N S T IY2 N M EH1 T R IH0 K IH1 Z EY1 D IH1 S T AH0 N S F AH1 NG K SH AH0 N D IH0 F AY1 N D B IY0 T W IY1 N P R AA2 B AH0 B IH1 L AH0 T IY0 D IH2 S T R AH0 B Y UW1 SH AH0 N Z AO1 N EY1 G IH1 V IH0 N M EH1 T R IH0 K S P EY1 S'
    ]
    data_list = list(text_encoder.encode(test) for test in tests)

    return data_list

def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()


    WaveGlow = get_WaveGlow('/kaggle/working/waveglow_256channels_ljs_v2.pt')

    data_list = get_data(text_encoder)
    for duration in [0.8, 1., 1.2]:
        for pitch in [0.8, 1., 1.2]:
            for energy in [0.8, 1., 1.2]:
                for i, phn in tqdm(enumerate(data_list)):
                    mel, mel_cuda = synthesis(model, phn, [duration, pitch, energy])
                    os.makedirs("results", exist_ok=True)

                    inference(
                        mel_cuda, WaveGlow,
                        f"/kaggle/working/results/d={duration}_p={pitch}_e={energy}_{i}_waveglow.wav"
                    )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=5,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
