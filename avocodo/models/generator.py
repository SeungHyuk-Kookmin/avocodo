import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from avocodo.models.utils import init_weights, get_padding


# Residual Block 정의
class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        # dilation convs (3개 dilation 값 사용)
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)   # 가중치 초기화

        # 후속 convs (dilation=1 고정)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        # residual connection 구조
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.2)   # pre-activation
            xt = c1(xt)                 # dilated conv
            xt = F.leaky_relu(xt, 0.2)
            xt = c2(xt)                 # 일반 conv
            x = xt + x                  # residual 연결
        return x

    def remove_weight_norm(self):
        # weight_norm 제거
        for _l in self.convs1:
            remove_weight_norm(_l)
        for _l in self.convs2:
            remove_weight_norm(_l)


# Generator 정의
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.resblock = h.resblock
        self.num_kernels = len(h.resblock_kernel_sizes)   # resblock 커널 개수
        self.num_upsamples = len(h.upsample_rates)        # upsample 단계 수

        # 입력 (mel spectrogram 80차원 → conv 입력 채널)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock

        # 업샘플링 블록 (ConvTranspose1d 여러 개)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            _ups = nn.ModuleList()
            for _i, (_u, _k) in enumerate(zip(u, k)):
                in_channel = h.upsample_initial_channel // (2**i)
                out_channel = h.upsample_initial_channel // (2**(i + 1))
                _ups.append(weight_norm(
                    ConvTranspose1d(in_channel, out_channel, _k, _u, padding=(_k - _u) // 2)))
            self.ups.append(_ups)

        # Residual Blocks & Post Convs
        self.resblocks = nn.ModuleList()
        self.conv_post = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = h.upsample_initial_channel // (2**(i + 1))
            temp = nn.ModuleList()
            # 여러 dilation 조합의 resblock 생성
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                temp.append(resblock(h, ch, k, d))
            self.resblocks.append(temp)

            # projection conv (sub-band 출력용)
            if self.h.projection_filters[i] != 0:
                self.conv_post.append(
                    weight_norm(
                        Conv1d(
                            ch, self.h.projection_filters[i],
                            self.h.projection_kernels[i], 1,
                            padding=self.h.projection_kernels[i] // 2
                        )))
            else:
                self.conv_post.append(torch.nn.Identity())

        # 가중치 초기화
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        outs = []               # multi-band 출력 저장
        x = self.conv_pre(x)    # 초기 conv

        # 업샘플링 + residual block 반복
        for i, (ups, resblocks, conv_post) in enumerate(zip(self.ups, self.resblocks, self.conv_post)):
            x = F.leaky_relu(x, 0.2)
            for _ups in ups:
                x = _ups(x)     # ConvTranspose로 upsampling

            xs = None
            # resblock 여러 개 적용 후 평균
            for j, resblock in enumerate(resblocks):
                if xs is None:
                    xs = resblock(x)
                else:
                    xs += resblock(x)
            x = xs / self.num_kernels

            # 마지막 3개 업샘플 단계에서만 실제 오디오 출력 생성
            if i >= (self.num_upsamples-3):
                _x = F.leaky_relu(x)
                _x = conv_post(_x)
                _x = torch.tanh(_x)   # [-1, 1] 범위로 스케일링
                outs.append(_x)
            else:
                # 중간 단계는 conv_post 통과 후 다음 단계로
                x = conv_post(x)

        return outs   # 멀티스케일 오디오 출력 리스트

    def remove_weight_norm(self):
        print('Removing weight norm...')
        # 업샘플링 weight_norm 제거
        for ups in self.ups:
            for _l in ups:
                remove_weight_norm(_l)
        # resblock 내부 weight_norm 제거
        for resblock in self.resblocks:
            for _l in resblock:
                _l.remove_weight_norm()
        # conv_pre weight_norm 제거
        remove_weight_norm(self.conv_pre)
        # conv_post weight_norm 제거
        for _l in self.conv_post:
            if not isinstance(_l, torch.nn.Identity):
                remove_weight_norm(_l)
