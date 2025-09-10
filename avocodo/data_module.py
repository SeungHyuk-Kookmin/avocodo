from dataclasses import dataclass
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from avocodo.meldataset import MelDataset
from avocodo.meldataset import get_dataset_filelist


# 데이터 설정을 담는 dataclass
@dataclass
class AvocodoDataConfig:
    segment_size: int        # 오디오 segment 길이
    num_mels: int            # Mel spectrogram 채널 수
    num_freq: int            # FFT frequency bins
    sampling_rate: int       # 샘플링 레이트
    n_fft: int               # FFT 크기
    hop_size: int            # hop size
    win_size: int            # window size
    fmin: int                # Mel 변환 시 최소 주파수
    fmax: int                # Mel 변환 시 최대 주파수
    batch_size: int          # 배치 크기
    num_workers: int         # DataLoader 병렬 워커 수

    fine_tuning: bool        # 파인튜닝 여부
    base_mels_path: str      # 사전 계산된 mel spectrogram 경로

    input_wavs_dir: str      # 원본 wav 경로
    input_mels_dir: str      # mel spectrogram 경로
    input_training_file: str # train 파일 리스트 txt
    input_validation_file: str # validation 파일 리스트 txt


# Avocodo용 DataModule
class AvocodoData(LightningDataModule):
    def __init__(self, h: AvocodoDataConfig):
        super().__init__()
        # Lightning 방식으로 하이퍼파라미터 저장 (hparams 속성으로 접근 가능)
        self.save_hyperparameters(h)

    def prepare_data(self):
        '''
        데이터 준비 단계
        - dataset 파일리스트(train, validation) 생성
        '''
        self.training_filelist, self.validation_filelist = get_dataset_filelist(
            self.hparams.input_wavs_dir,
            self.hparams.input_training_file,
            self.hparams.input_validation_file
        )

    def setup(self, stage=None):
        '''
        학습 세트 초기화
        - MelDataset 생성
        '''
        self.trainset = MelDataset(
            self.training_filelist,
            self.hparams.segment_size,
            self.hparams.n_fft,
            self.hparams.num_mels,
            self.hparams.hop_size,
            self.hparams.win_size,
            self.hparams.sampling_rate,
            self.hparams.fmin,
            self.hparams.fmax,
            n_cache_reuse=0,  # 캐시 사용 X
            fmax_loss=self.hparams.fmax_for_loss,  # loss 계산 시 fmax
            fine_tuning=self.hparams.fine_tuning,  # 파인튜닝 여부
            base_mels_path=self.hparams.input_mels_dir
        )

    def train_dataloader(self):
        '''
        학습용 DataLoader 반환
        '''
        return DataLoader(
            self.trainset,
            num_workers=self.hparams.num_workers,
            shuffle=False,                # ⚠ shuffle=False → 순차 학습 (보통 True를 씀)
            batch_size=self.hparams.batch_size,
            pin_memory=True,              # CUDA pinned memory
            drop_last=True                # 마지막 배치 버림 (정확한 배치 크기 유지)
        )

    @rank_zero_only
    def val_dataloader(self):
        '''
        검증용 DataLoader 반환
        - rank_zero_only: 분산 학습 시 0번 프로세스만 실행
        '''
        validset = MelDataset(
            self.validation_filelist,
            self.hparams.segment_size,
            self.hparams.n_fft,
            self.hparams.num_mels,
            self.hparams.hop_size,
            self.hparams.win_size,
            self.hparams.sampling_rate,
            self.hparams.fmin,
            self.hparams.fmax,
            False,     # augment 여부
            False,     # shuffle 여부
            n_cache_reuse=0,
            fmax_loss=self.hparams.fmax_for_loss,
            fine_tuning=self.hparams.fine_tuning,
            base_mels_path=self.hparams.input_mels_dir
        )
        return DataLoader(
            validset,
            num_workers=self.hparams.num_workers,
            shuffle=False,      # 검증 데이터는 섞지 않음
            sampler=None,       # 샘플러 없음
            batch_size=1,       # 검증은 1개씩
            pin_memory=True,
            drop_last=True
        )
