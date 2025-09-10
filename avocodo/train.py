import os
import argparse

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from avocodo.data_module import AvocodoData
from avocodo.lightning_module import Avocodo


# TensorBoard Logger 커스텀 (epoch 키 제거)
class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)   # epoch 항목 제거
        return super().log_metrics(metrics, step)   # 부모 클래스 메서드 호출


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 학습 인자들 정의
    parser.add_argument('--group_name', default=None)    # 실험 그룹 이름
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')   # 원본 wav 경로
    parser.add_argument('--input_mels_dir', default='ft_dataset')          # Mel 스펙트로그램 경로
    parser.add_argument('--input_training_file',
                        default='LJSpeech-1.1/training.txt')              # 학습용 파일 리스트
    parser.add_argument('--input_validation_file',
                        default='LJSpeech-1.1/validation.txt')            # 검증용 파일 리스트
    parser.add_argument('--config', default='avocodo/configs/avocodo_v1.json') # 설정 JSON 파일
    parser.add_argument('--training_epochs', default=5000, type=int)      # 학습 epoch 수
    parser.add_argument('--fine_tuning', default=False, type=bool)        # 파인튜닝 여부

    # 인자 파싱
    a = parser.parse_args()

    # OmegaConf에 커스텀 resolver 등록
    OmegaConf.register_new_resolver(
        "from_args", lambda x: getattr(a, x)   # CLI 인자에서 가져오기
    )
    OmegaConf.register_new_resolver(
        "dir", lambda base_dir, string: os.path.join(base_dir, string)   # 경로 join
    )

    # config 로드 및 resolve
    conf = OmegaConf.load(a.config)
    OmegaConf.resolve(conf)

    # 데이터 모듈 및 모델 초기화
    dm = AvocodoData(conf.data)
    model = Avocodo(conf.model)

    # 학습 제어 관련 설정
    limit_train_batches = 1.0   # 학습 데이터 전부 사용 (1.0 == 100%)
    limit_val_batches = 1.0     # 검증 데이터 전부 사용
    log_every_n_steps = 50      # 몇 스텝마다 로그 남길지
    max_epochs = conf.model.train.training_epochs  # 최대 epoch 수

    # PyTorch Lightning Trainer 정의
    trainer = Trainer(
        gpus=1,   # GPU 1개 사용
        max_epochs=max_epochs,   # 최대 학습 epoch
        callbacks=[
            # Rich 스타일 진행바 추가
            RichProgressBar(
                refresh_rate=1,   # 1 step마다 갱신
                theme=RichProgressBarTheme(
                    description="#AF81EB",
                    progress_bar="#8BE9FE",
                    progress_bar_finished="#8BE9FE",
                    progress_bar_pulse="#1363DF",
                    batch_progress="#AF81EB",
                    time="#1363DF",
                    processing_speed="#1363DF",
                    metrics="#9BF9FE",
                )
            )
        ],
        logger=TensorBoardLogger("logs", name="Avocodo"),  # 로그 저장 경로
        limit_train_batches=limit_train_batches,            # 학습 데이터 제한
        limit_val_batches=limit_val_batches,                # 검증 데이터 제한
        log_every_n_steps=log_every_n_steps                 # 로깅 주기
    )

    # 학습 시작
    trainer.fit(model, dm)
