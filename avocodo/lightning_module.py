import itertools

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from avocodo.meldataset import mel_spectrogram
from avocodo.models.generator import Generator
from avocodo.models.CoMBD import CoMBD
from avocodo.models.SBD import SBD
from avocodo.models.losses import feature_loss, generator_loss, discriminator_loss
from avocodo.pqmf import PQMF


# LightningModule: Generator + Discriminator + Losses
class Avocodo(LightningModule):
    def __init__(self, h):
        super().__init__()
        self.save_hyperparameters(h)   # config(hparams) 저장

        # PQMF (Polyphase Quadrature Mirror Filter) 
        # 멀티밴드 오디오 분해용 (Lv2, Lv1)
        self.pqmf_lv2 = PQMF(*self.hparams.pqmf_config["lv2"])
        self.pqmf_lv1 = PQMF(*self.hparams.pqmf_config["lv1"])

        # 모델 구성요소
        self.generator = Generator(self.hparams.generator)                  # Generator
        self.combd = CoMBD(self.hparams.combd, [self.pqmf_lv2, self.pqmf_lv1])  # CoMBD (Convolutional Multi-Band Discriminator)
        self.sbd = SBD(self.hparams.sbd)                                    # SBD (Sub-Band Discriminator)

    def configure_optimizers(self):
        # AdamW optimizer 2개 정의: Generator / Discriminator 각각
        h = self.hparams.optimizer
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            h.learning_rate,
            betas=[h.adam_b1, h.adam_b2]
        )
        opt_d = torch.optim.AdamW(
            itertools.chain(self.combd.parameters(), self.sbd.parameters()),
            h.learning_rate,
            betas=[h.adam_b1, h.adam_b2]
        )
        return [opt_g, opt_d], []   # Lightning이 G/D 두 optimizer를 관리

    def forward(self, z):
        # Generator의 최종 출력 (여러 scale 출력 중 마지막 것만 반환)
        return self.generator(z)[-1]

    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch 구성: (mel, waveform, file_id, mel_target)
        x, y, _, y_mel = batch
        y = y.unsqueeze(1)   # waveform shape: [B, 1, T]

        # PQMF로 분해한 real 오디오 (Lv2, Lv1, full-band)
        ys = [
            self.pqmf_lv2.analysis(y)[:, :self.hparams.generator.projection_filters[1]],
            self.pqmf_lv1.analysis(y)[:, :self.hparams.generator.projection_filters[2]],
            y
        ]

        # Generator forward (multi-scale outputs)
        y_g_hats = self.generator(x)

        # ===========================
        # (1) Generator 학습 단계
        # ===========================
        if optimizer_idx == 0:
            # CoMBD: real vs fake 판별
            y_du_hat_r, y_du_hat_g, fmap_u_r, fmap_u_g = self.combd(ys, y_g_hats)
            loss_fm_u, _ = feature_loss(fmap_u_r, fmap_u_g)   # Feature Matching Loss
            loss_gen_u, _ = generator_loss(y_du_hat_g)        # Generator Loss

            # SBD: real vs fake 판별
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.sbd(y, y_g_hats[-1])
            loss_fm_s, _ = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            # Mel L1 Loss
            y_g_hat_mel = mel_spectrogram(
                y_g_hats[-1].squeeze(1),
                self.hparams.audio.n_fft,
                self.hparams.audio.num_mels,
                self.hparams.audio.sampling_rate,
                self.hparams.audio.hop_size,
                self.hparams.audio.win_size,
                self.hparams.audio.fmin,
                self.hparams.audio.fmax_for_loss
            )
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
            self.log("train/l1_loss", loss_mel, prog_bar=True)
            loss_mel = loss_mel * self.hparams.loss_scale_mel

            # 총 Generator Loss = adversarial + FM + Mel loss
            g_loss = loss_gen_s + loss_gen_u + loss_fm_s + loss_fm_u + loss_mel
            self.log("train/g_loss", g_loss, prog_bar=True)
            loss = g_loss

        # ===========================
        # (2) Discriminator 학습 단계
        # ===========================
        if optimizer_idx == 1:
            # Generator 출력 detach (gradient 차단)
            detached_y_g_hats = [x.detach() for x in y_g_hats]

            # CoMBD Discriminator Loss
            y_du_hat_r, y_du_hat_g, _, _ = self.combd(ys, detached_y_g_hats)
            loss_disc_u, _, _ = discriminator_loss(y_du_hat_r, y_du_hat_g)

            # SBD Discriminator Loss
            y_ds_hat_r, y_ds_hat_g, _, _ = self.sbd(y, detached_y_g_hats[-1])
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # 총 Discriminator Loss
            d_loss = loss_disc_s + loss_disc_u
            self.log("train/d_loss", d_loss, prog_bar=True)
            loss = d_loss

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation: L1 Mel Loss 계산 + 오디오 로그 남김
        x, y, _, y_mel = batch
        y_g_hat = self(x)

        # Mel Spectrogram 비교
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            self.hparams.audio.n_fft,
            self.hparams.audio.num_mels,
            self.hparams.audio.sampling_rate,
            self.hparams.audio.hop_size,
            self.hparams.audio.win_size,
            self.hparams.audio.fmin,
            self.hparams.audio.fmax_for_loss
        )
        val_loss = F.l1_loss(y_mel, y_g_hat_mel)

        # Audio 로그 (TensorBoard에 waveform 저장)
        self.logger.experiment.add_audio(
            f'pred/{batch_idx}', y_g_hat.squeeze(), self.current_epoch, self.hparams.audio.sampling_rate
        )
        self.logger.experiment.add_audio(
            f'gt/{batch_idx}', y[0].squeeze(), self.current_epoch, self.hparams.audio.sampling_rate
        )
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        # Validation 결과 평균 후 로깅
        val_loss = torch.mean(torch.stack(validation_step_outputs))
        self.log("validation/l1_loss", val_loss, prog_bar=False)
