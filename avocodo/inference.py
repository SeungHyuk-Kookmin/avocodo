import os
import argparse
from omegaconf import OmegaConf
import torch
from scipy.io.wavfile import write

from pytorch_lightning import Trainer

from avocodo.meldataset import mel_spectrogram
from avocodo.meldataset import MAX_WAV_VALUE
from avocodo.meldataset import load_wav
from avocodo.meldataset import normalize
from avocodo.lightning_module import Avocodo
from avocodo.data_module import AvocodoData


h = None
device = None


# 멜 스펙트로그램 변환 함수
def get_mel(x):
    return mel_spectrogram(
        x,
        1024,   # n_fft
        80,     # num_mels
        22050,  # sampling rate
        256,    # hop length
        1024,   # win length
        0,      # fmin
        8000    # fmax
    )


# 추론 실행 함수
def inference(a, conf):
    # 저장된 checkpoint에서 LightningModule 로드
    avocodo = Avocodo.load_from_checkpoint(
        f"{a.checkpoint_path}/version_{a.version}/checkpoints/{a.checkpoint_file_id}",
        map_location='cpu'   # CPU에 먼저 로드
    )

    # 데이터 모듈 준비
    avocodo_data = AvocodoData(conf.audio)
    avocodo_data.prepare_data()
    validation_dataloader = avocodo_data.val_dataloader()

    # 출력 디렉토리 생성
    output_path = f'{a.output_dir}/version_{a.version}/'
    os.makedirs(output_path, exist_ok=True)

    # generator GPU로 이동 & weight_norm 제거
    avocodo.generator.to(a.device)
    avocodo.generator.remove_weight_norm()

    # TorchScript 변환 및 저장 (배포용 최적화)
    m = torch.jit.script(avocodo.generator)
    torch.jit.save(
        m,
        os.path.join(output_path, "scripted.pt")
    )

    # no_grad 모드 (추론만 실행, gradient 계산 X)
    with torch.no_grad():
        for i, batch in enumerate(validation_dataloader):
            mels, _, file_ids, _ = batch   # (mel, waveform, 파일명, 기타)
            
            # Generator를 통해 오디오 생성
            y_g_hat = avocodo(mels.to(a.device))

            # 배치 내 파일별로 저장
            for _y_g_hat, file_id in zip(y_g_hat, file_ids):
                audio = _y_g_hat.squeeze(0)               # [1, T] → [T]
                audio = audio * MAX_WAV_VALUE             # [-1,1] 범위 → [-32768, 32767]
                audio = audio.cpu().numpy().astype('int16')

                output_file = os.path.join(
                    output_path,
                    file_id.split('/')[-1]   # 원래 파일명 유지
                )
                print(file_id)   # 변환된 파일명 출력
                write(output_file, conf.audio.sampling_rate, audio)   # wav 저장
    print('Done inference')


def main():
    print('Initializing Inference Process..')

    # CLI 인자 정의
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='logs/Avocodo')  # 모델 로그 경로
    parser.add_argument('--version', type=int, required=True)         # Lightning 버전 (ex: version_0)
    parser.add_argument('--checkpoint_file_id', type=str, default='', required=True)  # checkpoint 파일명
    parser.add_argument('--output_dir', type=str, default='generated_files')  # 결과 저장 폴더
    parser.add_argument('--script', type=bool, default=True)          # TorchScript 저장 여부
    parser.add_argument('--device', type=str, default='cuda')         # 추론 장치
    a = parser.parse_args()

    # 저장된 hparams 불러오기
    conf = OmegaConf.load(os.path.join(a.checkpoint_path, f"version_{a.version}", "hparams.yaml"))

    # 추론 실행
    inference(a, conf)


if __name__ == '__main__':
    main()
