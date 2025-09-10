# -*- coding: utf-8 -*-
# PQMF (Pseudo QMF) 모듈
# Copyright 2020 Tomoki Hayashi
# MIT License
# 출처: ParallelWaveGAN (kan-bayashi)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import kaiser


# =============================
# Prototype Filter 설계
# =============================
def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """
    Kaiser 윈도우 기반 prototype filter 설계
    - taps: 필터 길이 (짝수여야 함)
    - cutoff_ratio: Nyquist 대비 컷오프 주파수 비율
    - beta: Kaiser window 계수
    
    참고: "A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks"
    """
    assert taps % 2 == 0, "The number of taps must be even."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be between 0 and 1."

    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        # Sinc 필터 형태
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    # 중앙 값(NaN 방지) → cutoff ratio로 대체
    h_i[taps // 2] = cutoff_ratio

    # Kaiser window 적용
    w = kaiser(taps + 1, beta)
    h = h_i * w
    return h   # 길이 (taps+1,) impulse response


# =============================
# PQMF 클래스
# =============================
class PQMF(torch.nn.Module):
    """
    Pseudo-QMF 모듈
    - Analysis: 입력 신호 → 서브밴드 분해
    - Synthesis: 서브밴드 신호 → 원 신호 복원
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """
        Args:
            subbands: 분해할 sub-band 개수
            taps: 필터 길이
            cutoff_ratio: 컷오프 주파수 비율
            beta: Kaiser window 계수
        """
        super(PQMF, self).__init__()

        # Prototype filter 생성
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)

        # Analysis/Synthesis 필터 초기화
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))

        # 각 subband k별 필터 계수 계산
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # torch tensor로 변환
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)   # (subbands, 1, taps+1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0) # (1, subbands, taps+1)

        # buffer 등록 (학습되지 않는 파라미터)
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # up/down 샘플링용 필터
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)

        self.subbands = subbands
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)   # conv padding

    # =============================
    # Analysis (분해)
    # =============================
    def analysis(self, x):
        """
        Args:
            x: 입력 (B, 1, T)
        Returns:
            분해된 서브밴드 (B, subbands, T // subbands)
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)                # analysis filter 적용
        return F.conv1d(x, self.updown_filter, stride=self.subbands)      # 다운샘플링 (stride=subbands)

    # =============================
    # Synthesis (합성)
    # =============================
    def synthesis(self, x):
        """
        Args:
            x: 서브밴드 입력 (B, subbands, T // subbands)
        Returns:
            합성된 waveform (B, 1, T)
        """
        # ConvTranspose로 업샘플링 (stride=subbands)
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        # Synthesis filter 적용
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
