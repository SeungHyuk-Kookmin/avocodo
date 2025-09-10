from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm
from typing import List

from avocodo.pqmf import PQMF
from avocodo.models.utils import get_padding


# CoMBDBlock: Convolutional Multi-Band Discriminator Block
class CoMBDBlock(torch.nn.Module):
    def __init__(
        self,
        h_u: List[int],   # hidden units (레이어별 채널 수)
        d_k: List[int],   # kernel sizes
        d_s: List[int],   # strides
        d_d: List[int],   # dilations
        d_g: List[int],   # groups
        d_p: List[int],   # paddings
        op_f: int,        # projection conv 출력 채널 수
        op_k: int,        # projection conv 커널 사이즈
        op_g: int,        # projection conv 그룹 수
        use_spectral_norm=False  # spectral_norm 쓸지 여부
    ):
        super(CoMBDBlock, self).__init__()
        # weight_norm 또는 spectral_norm 적용
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        # conv 블록들 저장
        self.convs = nn.ModuleList()
        # 입력 → 첫 번째 hidden 채널 연결
        filters = [[1, h_u[0]]]
        # 이후 hidden 채널 쌍 정의
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])
        # 각 conv 레이어 생성
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(norm_f(
                Conv1d(
                    in_channels=_f[0],
                    out_channels=_f[1],
                    kernel_size=_k,
                    stride=_s,
                    dilation=_d,
                    groups=_g,
                    padding=_p
                )
            ))
        # projection conv (마지막 출력 레이어)
        self.projection_conv = norm_f(
            Conv1d(
                in_channels=filters[-1][1],  # 마지막 hidden 채널 수
                out_channels=op_f,
                kernel_size=op_k,
                groups=op_g
            )
        )

    def forward(self, x):
        fmap = []   # feature map 저장용
        for block in self.convs:
            x = block(x)                # conv 통과
            x = F.leaky_relu(x, 0.2)    # LeakyReLU 활성화
            fmap.append(x)              # feature map 기록
        x = self.projection_conv(x)     # projection conv 적용
        return x, fmap                  # 출력, feature maps 반환


# CoMBD: Convolutional Multi-Band Discriminator
class CoMBD(torch.nn.Module):
    def __init__(self, h, pqmf_list=None, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        self.h = h
        # PQMF(Polyphase Quadrature Mirror Filter) 설정
        if pqmf_list is not None:
            self.pqmf = pqmf_list
        else:
            self.pqmf = [
                PQMF(*h.pqmf_config["lv2"]),  # 2레벨 PQMF
                PQMF(*h.pqmf_config["lv1"])   # 1레벨 PQMF
            ]

        # discriminator 블록들을 모듈리스트에 저장
        self.blocks = nn.ModuleList()
        for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
            h.combd_h_u,  # hidden units
            h.combd_d_k,  # kernel sizes
            h.combd_d_s,  # strides
            h.combd_d_d,  # dilations
            h.combd_d_g,  # groups
            h.combd_d_p,  # paddings
            h.combd_op_f, # projection out channels
            h.combd_op_k, # projection kernel size
            h.combd_op_g, # projection groups
        ):
            self.blocks.append(CoMBDBlock(
                _h_u,
                _d_k,
                _d_s,
                _d_d,
                _d_g,
                _d_p,
                _op_f,
                _op_k,
                _op_g,
            ))

    # 여러 블록 forward helper
    def _block_forward(self, input, blocks, outs, f_maps):
        for x, block in zip(input, blocks):
            out, f_map = block(x)
            outs.append(out)       # discriminator 출력 추가
            f_maps.append(f_map)   # feature map 추가
        return outs, f_maps

    # PQMF 기반 멀티스케일 forward
    def _pqmf_forward(self, ys, ys_hat):
        # 실제(real) / 예측(fake) 오디오 입력을 PQMF로 분해
        multi_scale_inputs = []
        multi_scale_inputs_hat = []
        for pqmf in self.pqmf:
            multi_scale_inputs.append(
                pqmf.to(ys[-1]).analysis(ys[-1])[:, :1, :]  # real audio
            )
            multi_scale_inputs_hat.append(
                pqmf.to(ys[-1]).analysis(ys_hat[-1])[:, :1, :]  # generated audio
            )

        outs_real = []
        f_maps_real = []
        # 실제 오디오 - 계층적 forward
        outs_real, f_maps_real = self._block_forward(
            ys, self.blocks, outs_real, f_maps_real)
        # 실제 오디오 - 멀티스케일 forward
        outs_real, f_maps_real = self._block_forward(
            multi_scale_inputs, self.blocks[:-1], outs_real, f_maps_real)

        outs_fake = []
        f_maps_fake = []
        # 예측 오디오 - 계층적 forward
        outs_fake, f_maps_fake = self._block_forward(
            ys_hat, self.blocks, outs_fake, f_maps_fake)
        # 예측 오디오 - 멀티스케일 forward
        outs_fake, f_maps_fake = self._block_forward(
            multi_scale_inputs_hat, self.blocks[:-1], outs_fake, f_maps_fake)

        # real / fake 판별 결과 + feature map 반환
        return outs_real, outs_fake, f_maps_real, f_maps_fake

    # 최종 forward
    def forward(self, ys, ys_hat):
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(
            ys, ys_hat)
        return outs_real, outs_fake, f_maps_real, f_maps_fake
