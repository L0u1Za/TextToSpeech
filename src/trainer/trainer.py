import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.model.vocoder import Vocoder
from src.base import BaseTrainer
from src.base.base_text_encoder import BaseTextEncoder
from src.logger.utils import plot_spectrogram_to_buf
from src.metric.utils import calc_cer, calc_wer
from src.utils import inf_loop, MetricTracker, get_WaveGlow

import waveglow

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            text_encoder,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            *criterion.names, "total_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "total_loss", *[m.name for m in self.metrics], writer=self.writer
        )

        if self.config["vocoder"]:
            #self.vocoder = Vocoder(self.config["vocoder"]["path"])
            #self.vocoder = self.vocoder.to(device)
            #self.vocoder.eval()
            self.vocoder = get_WaveGlow(self.config["vocoder"]["path"])
            self.vocoder = self.vocoder.to(device).eval()

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "phone_encoded", "src_pos", "mel_pos", "phone_duration", "pitch", "energy"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["total_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_spectrogram(batch["pred_mel"], "predicted_spectrogram")
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["pred_mel"] = outputs[0]
            batch["predicted"] = outputs[1]

        batch["predictor_targets"] = [batch["phone_duration"], batch["pitch"], batch["energy"]]

        batch["loss"] = self.criterion(**batch)
        final_loss = torch.tensor(0.0).to(self.device)
        for loss in batch["loss"]:
            final_loss += loss
        batch["total_loss"] = final_loss

        if is_train:
            final_loss.backward()

            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            for i, loss in enumerate(batch["loss"]):
                metrics.update(self.criterion.names[i], loss.item())

        metrics.update("total_loss", final_loss.item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            #self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])
            self._log_spectrogram(batch["pred_mel"], "predicted_spectrogram")

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            pred_mel,
            phone,
            predicted,
            predictor_targets,
            audio_path,
            pred_audio,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        tuples = list(zip(pred_mel, text, phone, *predicted, *predictor_targets, audio_path))
        shuffle(tuples)
        rows = {}
        for p_mel, txt, ph, duration_pred, pitch_pred, energy_pred, duration, pitch, energy, audio_path in tuples[:examples_to_log]:
            d_p = ' '.join([str(s) for s in duration_pred.squeeze(). tolist()])
            p_p = ' '.join([str(s) for s in pitch_pred.squeeze().tolist()])
            e_p = ' '.join([str(s) for s in energy_pred.squeeze().tolist()])
            d = ' '.join([str(s) for s in duration.squeeze().tolist()])
            p = ' '.join([str(s) for s in pitch.squeeze().tolist()])
            e = ' '.join([str(s) for s in energy.squeeze().tolist()])

            audio = waveglow.inference.get_wav(p_mel, self.vocoder)

            rows[Path(audio_path).name] = {
                "text": txt,
                "phoneme": ph,
                "duration_prediction": d_p,
                "pitch_predicion": p_p,
                "energy_prediction": e_p,
                "duration_target": d,
                "pitch_target": p,
                "energy_target": e,
                "pred_audio": self.writer.wandb.Audio(audio, sample_rate=self.config["preprocessing"]["sr"])
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch, spec_name="spectrogram"):
        spectrogram = random.choice(spectrogram_batch.detach().cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(spec_name, ToTensor()(image))

    def _log_audio(self, audio_batch, name, sr=22500):
        audio = random.choice(audio_batch.detach().cpu())
        self.writer.add_audio(name, audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
