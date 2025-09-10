# -*- coding: utf-8 -*-
# Copyright 2020 Jungil Kong
# MIT License
"""
HiFi-GAN에서 가져온 loss 함수 모듈
"""

import torch


# Feature Matching Loss (FM Loss)
def feature_loss(fmap_r, fmap_g):
    """
    fmap_r: real 오디오에서 얻은 feature maps
    fmap_g: generator가 만든 fake 오디오에서 얻은 feature maps
    
    Discriminator의 중간 feature map을 비교해서
    Generator가 real-like한 특징을 학습하게 만듦.
    """
    loss = 0
    losses = []
    for dr, dg in zip(fmap_r, fmap_g):   # 각 discriminator 층별 feature map
        for rl, gl in zip(dr, dg):       # layer별 feature map
            _loss = torch.mean(torch.abs(rl - gl))  # L1 거리
            loss += _loss
        losses.append(_loss)             # 마지막 layer loss만 저장됨 (원본 코드 구조 그대로)

    return loss*2, losses   # 전체 loss는 2배 가중치


# Discriminator Loss (LSGAN 방식)
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    disc_real_outputs: discriminator(real audio 입력)
    disc_generated_outputs: discriminator(fake audio 입력)

    LSGAN 방식을 사용:
      - real → (1 - D(x))^2 최소화
      - fake → (D(G(z)))^2 최소화
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)   # real은 1에 가깝게
        g_loss = torch.mean(dg**2)       # fake는 0에 가깝게
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())   # 로깅용
        g_losses.append(g_loss.item())   # 로깅용

    return loss, r_losses, g_losses


# Generator Loss (LSGAN 방식)
def generator_loss(disc_outputs):
    """
    disc_outputs: discriminator가 fake audio를 판별한 출력
    
    Generator 입장에서는 fake도 real처럼 보이길 원하므로:
      - (1 - D(G(z)))^2 최소화
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)   # layer별 loss 기록
        loss += l

    return loss, gen_losses
