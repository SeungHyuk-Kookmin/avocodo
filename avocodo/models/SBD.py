from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm

from avocodo.pqmf import PQMF
from avocodo.models.utils import get_padding


# MDC: Multi-Dilation Convolution
class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,       # 입력 채널 수
        out_channels,      # 출력 채널 수
        strides,           # stride
        kernel_size,       # 커널 사이즈 리스트
        dilations,         # dilation 리스트
        use_spectral_norm=False
    ):
        super(MDC, self).__init__()
        # spectral_norm / weight_norm 중 선택
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        # dilation 별 conv 레이어 여러 개 생성
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_k,
                    dilation=_d,
                    padding=get_padding(_k, _d)  # dilation에 맞춘 padding
                ))
            )
        # 이후 통합 conv
        self.post_conv = norm_f(Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=strides,
            padding=get_padding(_k, _d)  # 마지막 dilation 기준 padding
        ))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        # 여러 dilation conv 실행
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)  # (B, C, T) → (B, C, T, 1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)  # 여러 dilation 결과 concat
        x = torch.sum(_out, dim=-1)          # dilation별 출력 합산
        x = self.post_conv(x)                # post conv
        x = F.leaky_relu(x, 0.2)             # 활성화

        return x


# SBDBlock: Sub-Band Discriminator Block
class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,        # 입력 segment 크기
        strides,            # stride 리스트
        filters,            # 채널 수 리스트
        kernel_size,        # 커널 사이즈 리스트
        dilations,          # dilation 리스트
        use_spectral_norm=False
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        # (입력 → 첫 conv) + (중간 conv들) 채널 매핑 정의
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        # 각 레이어에 MDC 사용
        for _s, _f, _k, _d in zip(
            strides,
            filters_in_out,
            kernel_size,
            dilations
        ):
            self.convs.append(MDC(
                in_channels=_f[0],
                out_channels=_f[1],
                strides=_s,
                kernel_size=_k,
                dilations=_d,
                use_spectral_norm=use_spectral_norm
            ))
        # 마지막 projection conv
        self.post_conv = norm_f(Conv1d(
            in_channels=_f[1],   # 마지막 필터 채널
            out_channels=1,      # 스칼라 출력
            kernel_size=3,
            stride=1,
            padding=3 // 2       # 동일 패딩
        ))

    def forward(self, x):
        fmap = []   # feature map 저장
        for _l in self.convs:
            x = _l(x)
            fmap.append(x)      # 각 conv 출력 저장
        x = self.post_conv(x)   # 최종 conv
        return x, fmap


# Config wrapper (모델 설정 정리)
class MDCDConfig:
    def __init__(self, h):
        self.pqmf_params = h.pqmf_config["sbd"]     # 기본 PQMF
        self.f_pqmf_params = h.pqmf_config["fsbd"]  # fine PQMF
        self.filters = h.sbd_filters
        self.kernel_sizes = h.sbd_kernel_sizes
        self.dilations = h.sbd_dilations
        self.strides = h.sbd_strides
        self.band_ranges = h.sbd_band_ranges
        self.transpose = h.sbd_transpose
        self.segment_size = h.segment_size


# SBD: Sub-Band Discriminator
class SBD(torch.nn.Module):
    def __init__(self, h, use_spectral_norm=False):
        super(SBD, self).__init__()
        self.config = MDCDConfig(h)
        # PQMF 분석 필터
        self.pqmf = PQMF(
            *self.config.pqmf_params
        )
        # fine PQMF (transpose 플래그에 따라 적용)
        if True in h.sbd_transpose:
            self.f_pqmf = PQMF(
                *self.config.f_pqmf_params
            )
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        # 각 sub-band별 discriminator block 생성
        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose
        ):
            # transpose 여부에 따라 segment dimension 계산
            if _tr:
                segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            self.discriminators.append(SBDBlock(
                segment_dim=segment_dim,
                filters=_f,
                kernel_size=_k,
                dilations=_d,
                strides=_s,
                use_spectral_norm=use_spectral_norm
            ))

    def forward(self, y, y_hat):
        y_d_rs = []   # real outputs
        y_d_gs = []   # fake outputs
        fmap_rs = []  # real feature maps
        fmap_gs = []  # fake feature maps

        # PQMF로 real/fake 모두 sub-band 변환
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        # sub-band discriminator별 forward
        for d, br, tr in zip(
            self.discriminators,
            self.config.band_ranges,
            self.config.transpose
        ):
            if tr:  # transpose 적용 시
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            else:   # 일반 sub-band
                _y_in = y_in[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in[:, br[0]:br[1], :]
            # discriminator 실행
            y_d_r, fmap_r = d(_y_in)
            y_d_g, fmap_g = d(_y_hat_in)
            # 결과 저장
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
