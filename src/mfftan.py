import torch
import torch.nn as nn
import torch.nn.functional as F

'''
在encoder, decoder中添加skff机制
'''


def make_model(args, parent=False):
    model = MSCAN(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return model


class LocalGlobalChannelAttentionBlock(nn.Module):
    def __init__(self, n_feats, reduction=8):
        super(LocalGlobalChannelAttentionBlock, self).__init__()

        self.local_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
            nn.SELU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0)
        )

        self.global_att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
            nn.SELU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        local_att = self.local_att(x)
        global_att = self.global_att(x)
        att = self.sigmoid(local_att + global_att)
        out = x * att
        return out


#####################################################
class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, 1, 0, bias=bias),
            nn.PReLU()
        )
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, 1, 1, 0, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return feats_V


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        res = self.body(x)
        return res + x


'''
1)使用7x7卷积核扩展感受野, 深度可分离卷积，减少计算量
2）使用1x1卷积核扩充通道数，inverted bottleneck
# TODO 把通道数从64 --> 96
'''


class ConvNeXtBlock(nn.Module):
    def __init__(self, n_feats=64, kernel_size=7, expand_rate=1.5):
        super(ConvNeXtBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats * expand_rate, 1, 1, 0),
            # nn.GELU(),
            nn.PReLU(),
            nn.Conv2d(n_feats * expand_rate, n_feats, 1, 1, 0)
        )

    def forward(self, x):
        res = self.body(x)
        return res + x


class FFTResidualBlock(nn.Module):
    def __init__(self, n_feats, norm='backward'):
        self.norm = norm
        super(FFTResidualBlock, self).__init__()
        self.main_branch = nn.Sequential(
            ResBlock(n_feats),
            # ConvNeXtBlock(n_feats, kernel_size=7, expand_rate=1.5)
            nn.Conv2d(n_feats, n_feats, kernel_size=7, stride=1, padding=7 // 2, groups=n_feats)
        )
        # self.lgca = LocalGlobalChannelAttentionBlock(n_feats)

        self.fft_branch = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(n_feats * 2, n_feats * 2, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.fft_branch(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main_branch(x) + y + x


class EBlock(nn.Module):
    def __init__(self, n_feats=64, n_bloks=3):
        super(EBlock, self).__init__()
        layers = [FFTResidualBlock(n_feats) for _ in range(n_bloks)]
        self.layers = nn.Sequential(*layers)
        self.lgcab = LocalGlobalChannelAttentionBlock(n_feats)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x):
        res = self.layers(x)
        res = self.lgcab(res)
        # TODO 这里是否添加残差
        return res + self.alpha * x


class Dblock(nn.Module):
    def __init__(self, n_feats=64, n_blocks=3):
        super(Dblock, self).__init__()
        layers = [FFTResidualBlock(n_feats) for _ in range(n_blocks)]
        self.layers = nn.Sequential(*layers)
        self.lgcab = LocalGlobalChannelAttentionBlock(n_feats)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x):
        res = self.layers(x)
        res = self.lgcab(res)
        return res + self.alpha * x


class SCM(nn.Module):
    def __init__(self, in_channels, n_feats):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        res = self.main(x)
        return res


# Multi-Scale Channel Attention Network for Remote Sensing Image Super-Resolution
class MSCAN(nn.Module):
    def __init__(self, args):
        super(MSCAN, self).__init__()
        n_feats = args.n_feats
        n_blocks = [3, 2, 1]
        scale = args.scale[0]

        self.feature_extract1 = SCM(in_channels=3, n_feats=n_feats)
        self.e1 = EBlock(n_feats, n_blocks[0])
        # 下采样的方式 插值，步长卷积 还是 unpixelshuffle
        self.down1 = nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, stride=2, padding=1)

        self.feature_extract2 = SCM(in_channels=12, n_feats=n_feats * 2)
        self.skff_e_2 = SKFF(in_channels=n_feats * 2, height=2)
        self.e2 = EBlock(n_feats * 2, n_blocks[1])
        self.down2 = nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, stride=2, padding=1)

        self.feature_extract3 = SCM(in_channels=48, n_feats=n_feats * 4)
        self.skff_e_3 = SKFF(n_feats * 4, height=2)
        self.e3 = EBlock(n_feats * 4, n_blocks[2])

        # self.skff_d_3 = SKFF(in_channels=)
        self.down1_3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, stride=2, padding=1)
        )
        self.down2_3 = nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, stride=2, padding=1)
        self.skff_d_3 = SKFF(in_channels=n_feats * 4, height=3, )
        self.d3 = Dblock(n_feats * 4, n_blocks[2])
        self.up3 = nn.ConvTranspose2d(n_feats * 4, n_feats * 2, kernel_size=4, padding=1, stride=2)

        self.down1_2 = nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, stride=2, padding=1)
        self.up3_2 = nn.ConvTranspose2d(n_feats * 4, n_feats * 2, kernel_size=4, padding=1, stride=2)
        self.skff_d_2 = SKFF(in_channels=n_feats * 2, height=3)
        self.feature_fuse2 = nn.Conv2d(n_feats * 4, n_feats * 2, 1, 1, 0)
        self.d2 = Dblock(n_feats * 2, n_blocks[1])
        self.up2 = nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=4, padding=1, stride=2)

        self.up3_1 = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 4, n_feats * 2, kernel_size=4, padding=1, stride=2),
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=4, padding=1, stride=2)
        )
        self.up2_1 = nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=4, padding=1, stride=2)
        self.skff_d_1 = SKFF(in_channels=n_feats, height=3)
        self.feature_fuse1 = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)
        self.d1 = Dblock(n_feats, n_blocks[0])

        self.reconstruct = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

        self.alpha = nn.Parameter(torch.FloatTensor([1.]))

    def forward(self, x):
        x2 = nn.PixelUnshuffle(downscale_factor=2)(x)
        x4 = nn.PixelUnshuffle(downscale_factor=4)(x)

        f1 = self.feature_extract1(x)
        e1 = self.e1(f1)
        down1 = self.down1(e1)

        e2 = self.e2(self.skff_e_2([self.feature_extract2(x2), down1]))
        down2 = self.down2(e2)

        e3 = self.e3(self.skff_e_3([self.feature_extract3(x4), down2]))

        down1_3 = self.down1_3(e1)
        down2_3 = self.down2_3(e2)
        d3 = self.d3(self.skff_d_3([down1_3, down2_3, e3]))

        up3 = self.up3(d3)
        up3_2 = self.up3_2(e3)
        down1_2 = self.down1_2(e1)

        skff_d_2 = self.skff_d_2([up3_2, down1_2, e2])
        d2 = self.d2(self.feature_fuse2(torch.cat([up3, skff_d_2], dim=1)))
        up2 = self.up2(d2)

        skff_d_1 = self.skff_d_1([self.up3_1(e3), self.up2_1(e2), e1])
        d1 = self.d1(self.feature_fuse1(torch.cat([up2, skff_d_1], dim=1)))
        out = self.reconstruct(d1 + self.alpha * f1)
        return out


if __name__ == '__main__':
    from option import args
    from thop import profile

    a = torch.Tensor(1, 3, 48, 48)
    model = MSCAN(args)
    total_params = sum(p.numel() for p in model.parameters())
    flops, params = profile(model, inputs=(a,))
    print(total_params)
    print(flops / 1e9, params)
