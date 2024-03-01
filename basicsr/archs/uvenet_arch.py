import torch
from torch import nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


from basicsr.archs.arch_util import trunc_normal_, DropPath, PixelShuffleUpsample
from basicsr.utils.registry import ARCH_REGISTRY


ARCH_SETTINGS = {
    'tiny': {
        'depths': [3, 3, 9, 3],
        'channels': [96, 192, 384, 768]
    },
    'small': {
        'depths': [3, 3, 27, 3],
        'channels': [96, 192, 384, 768]
    },
    'base': {
        'depths': [3, 3, 27, 3],
        'channels': [128, 256, 512, 1024]
    },
    'large': {
        'depths': [3, 3, 27, 3],
        'channels': [192, 384, 768, 1536]
    },
    'xlarge': {
        'depths': [3, 3, 27, 3],
        'channels': [256, 512, 1024, 2048]
    },
}


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.InstanceNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """


    def __init__(self, in_chans=3, arch_type='tiny', drop_path_rate=0., layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3]):
        super().__init__()

        depths = ARCH_SETTINGS[arch_type]['depths']
        dims = ARCH_SETTINGS[arch_type]['channels']

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.InstanceNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.InstanceNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = out_indices

        for i in self.out_indices:
            norm_layer = nn.InstanceNorm2d(dims[i])
            self.add_module(f'norm{i}', norm_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # The output of LayerNorm2d may be discontiguous, which may cause some problem in the downstream tasks
                outs.append(norm_layer(x).contiguous())

        return outs


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(self.avg_pool(x)))
        out = attn * x
        return out


# Global Restoration Module
class GRM(nn.Module):
    def __init__(self, frames, n_feat=32):
        super().__init__()

        self.attention_module = nn.Sequential(
            nn.Conv2d(frames*3+3, n_feat, 3, 1, 1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False),
            nn.Conv2d(n_feat, 3, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, all_frames, rgb_x):
        _, frames = all_frames.shape[:2]
        all_frames = rearrange(all_frames, 'b t c h w -> b (t c) h w')
        conf_in = torch.cat([all_frames, rgb_x], dim=1)
        conf_out = self.attention_module(conf_in)
        out = conf_out * rgb_x
        return out


# Feature Alignment and Aggregation Module
class FAAM(nn.Module):
    def __init__(self, s, n_feat, frames, shift1=True):
        super().__init__()
        self.s = s
        self.n_feat = n_feat
        self.shift1 = shift1
        self.agg_layer = nn.Sequential(
            DepthwiseSeparableConv(n_feat*frames, n_feat),
            ChannelAttention(n_feat)
        )

    def spatial_shift(self, hw):
        frames = hw.shape[1]
        s = self.s
        n = self.n_feat // 8

        s_out = torch.zeros_like(hw)
        s_out[:,:,0*n:1*n,s:,s:] = hw[:,:,0*n:1*n,:-s,:-s]
        s_out[:,:,1*n:2*n,s:,0:] = hw[:,:,1*n:2*n,:-s,:]
        s_out[:,:,2*n:3*n,s:,0:-s] = hw[:,:,2*n:3*n,:-s,s:]
        s_out[:,:,3*n:4*n,:,s:] = hw[:,:,3*n:4*n,:,:-s]
        s_out[:,:,4*n:5*n,:,0:-s] = hw[:,:,4*n:5*n,:,s:]
        s_out[:,:,5*n:6*n,0:-s,s:] = hw[:,:,5*n:6*n,s:,:-s]
        s_out[:,:,6*n:7*n,0:-s,0:] = hw[:,:,6*n:7*n,s:,:]
        s_out[:,:,7*n:8*n,0:-s,0:-s] = hw[:,:,7*n:8*n,s:,s:]

        s_out[:,frames//2,:,:,:] = hw[:,frames//2,:,:,:]
        return s_out

    def forward(self, x):
        # x: (B,T,C,H,W)
        batch_size, frames = x.shape[:2]
        shortcut = x[:, frames//2, ...]
        shift_x = self.spatial_shift(x)
        shift_x = rearrange(shift_x, 'b t c h w -> b (t c) h w')

        out = self.agg_layer(shift_x)  # b (t c) h w -> b c h w

        # out = rearrange(shift_x, 'b n c h w -> b (n c) h w')

        return out + shortcut  # out: (B,C,H,W)


class Upsampler(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 num_feat=64,
                 num_out_ch=3):
        super().__init__()
        self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        self.upsample1 = PixelShuffleUpsample(2, num_feat)
        self.upsample2 = PixelShuffleUpsample(2, num_feat)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        nn.init.constant_(self.conv_last.weight, 0)
        nn.init.constant_(self.conv_last.bias, 0)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        hr = self.lrelu(self.conv_before_upsample(x))
        hr = self.lrelu(self.upsample1(hr))
        hr = self.lrelu(self.upsample2(hr))
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)

        return hr


class Decoder(nn.Module):
    def __init__(self, arch_type='tiny'):
        super().__init__()
        dims = ARCH_SETTINGS[arch_type]['channels']
        self.up1 = nn.Conv2d(dims[0], dims[0], 1, stride=1, padding=0, bias=False)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[1], dims[0], 1, stride=1, padding=0, bias=False)
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[2], dims[0], 1, stride=1, padding=0, bias=False)
            )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[3], dims[0], 1, stride=1, padding=0, bias=False)
            )
        self.up_last = Upsampler(4*dims[0], dims[0], 3)

    def forward(self, x):
        # x: [(B,C1,H/4,W/4), (B,C2,H/8,W/8), (B,C3,H/16,W/16), (B,C4,H/32,W/32)]
        x_all = torch.cat([
            self.up1(x[0]),
            self.up2(x[1]),
            self.up3(x[2]),
            self.up4(x[3])
            ], dim=1)
        out = self.up_last(x_all)

        return out  # out: (B,3,H,W)


@ARCH_REGISTRY.register()
class UVENet(nn.Module):
    def __init__(self,
                 arch_type='tiny',
                 num_frame=5,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3]
                 ):
        super().__init__()

        assert out_indices == [0,1,2,3]

        self.backbone = ConvNeXt(3, arch_type, drop_path_rate, layer_scale_init_value, out_indices)
        self.aams = nn.ModuleList([
            FAAM(s=3, n_feat=96, frames=num_frame),
            FAAM(s=3, n_feat=192, frames=num_frame),
            FAAM(s=3, n_feat=384, frames=num_frame),
            FAAM(s=3, n_feat=768, frames=num_frame),
        ])
        self.decoder = Decoder(arch_type)

        self.ga = GRM(num_frame, 64)

    def forward(self, x):
        # x: (B,T,3,H,W)
        batch_size, frames = x.shape[:2]
        shortcut = x

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        ms_feats = self.backbone(x)  # [(B*T,C1,H/4,W/4), (B*T,C2,H/8,W/8), (B*T,C3,H/16,W/16), (B*T,C4,H/32,W/32)]

        ms_outs = []
        for i, ss_feat in enumerate(ms_feats):
            ss_feat = rearrange(ss_feat, '(b t) c h w -> b t c h w', b=batch_size)
            ss_out = self.aams[i](ss_feat)
            ms_outs.append(ss_out)

        out = torch.sigmoid(self.decoder(ms_outs))
        out = self.ga(shortcut, out)

        return out  # (B,3,H,W)


# if __name__ == '__main__':
    # model = ConvNeXt().cuda()
    # input = torch.rand(2, 3, 256, 256).cuda()
    # output = model(input)
    # print('===================')
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)
    # torch.Size([2, 96, 64, 64])
    # torch.Size([2, 192, 32, 32])
    # torch.Size([2, 384, 16, 16])
    # torch.Size([2, 768, 8, 8])

