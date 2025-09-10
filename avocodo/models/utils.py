# -*- coding: utf-8 -*-
# Copyright 2020 Jungil Kong
# MIT License
"""
HiFi-GAN util 함수 모음
"""


def get_padding(kernel_size, dilation=1):
    """
    Conv1d / Conv2d 등에서 padding 값을 자동 계산하는 함수
    - kernel_size: 커널 크기
    - dilation: dilation factor

    공식: padding = ((kernel_size * dilation) - dilation) / 2
      → receptive field가 중앙 정렬되도록 맞추는 역할
    """
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    """
    Conv 레이어 초기화 함수
    - m: layer 모듈
    - mean: 평균
    - std: 표준편차

    Conv 계열(layer 이름에 "Conv"가 들어가는 경우) weight를
    N(mean, std^2) 정규분포로 초기화.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
