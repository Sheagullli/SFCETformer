# This network is designed based on single self-attention and multi-scale frequency block cross learning transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Fourier
class FourierUnit(nn.Module):

    def __init__(self, dim):
        # bn_layer not used
        super(FourierUnit, self).__init__()

        hidden_features = int(dim)

        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))
    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]

    def forward(self, x):
        x,pad_w = self.pad(x,2)
        x = torch.fft.rfft2(x)
        x = self.fft_channel_weight * x + self.fft_channel_bias
        x = torch.fft.irfft2(x)
        output = self.unpad(x,pad_w)

        return output

################################### ###################################################################################
# Multi-Scale frequency branch
class MFrequencyBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(MFrequencyBlock, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.conv_init = nn.Conv2d(dim, hidden_features * 2, 1)

        self.dwconv1_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3 // 2,
                      groups=hidden_features, padding_mode='reflect')

        self.dwconv1_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=5// 2,
                      groups=hidden_features, padding_mode='reflect')

        self.FFC = FourierUnit(hidden_features)
        self.conv_fina =  nn.Conv2d(hidden_features * 3, dim, 1)

    def forward(self, x):
        x_initial = self.conv_init(x)
        x1,x2 = x_initial.chunk(2, dim=1)

        x1 = self.dwconv1_1(x1)

        x2 = self.dwconv1_2(x2)

        # 此处对x1, x2, x3, x4分别做傅里叶变换及反变换
        x1 = self.FFC(x1)
        x2 = self.FFC(x2)

        x_combine = torch.cat([x1, x2, x], dim=1)

        output = self.conv_fina(x_combine)

        return output



##########################################################################
# multi-scale block (MSblokc)
class MSblock(nn.Module):
    def __init__ (self, dim, bias):
        super (MSblock, self).__init__()

        hidden_features = dim

        self.conv_inital = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias) #初始化

        self.dwconv1_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3 // 2,
                       groups= hidden_features, padding_mode='reflect')

        self.dwconv1_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=5 // 2,
                       groups=hidden_features, padding_mode='reflect')


        self.conv_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        xinital = self.conv_inital(x)
        x1,x2 = xinital.chunk(2, dim=1)
        y1, y2 = x1, x2

        x1 = self.conv(x1)
        x1 =  y1 * torch.sigmoid(x1)

        x2 = self.dwconv1_1(x2)
        x2 = x2 * torch.sigmoid(x1)

        x_combine = torch.cat([x1, x2], dim=1)

        x_out = self.conv_out(x_combine)

        return x_out


##########################################################################
# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_1 = nn.Conv2d(hidden_features , hidden_features , kernel_size=3, stride=1, padding=1,
                                groups=hidden_features , bias=bias)
        self.dwconv_2=nn.Conv2d(hidden_features , hidden_features , kernel_size=5, padding='same',
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.fft_channel_weight_1 = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))
        self.fft_channel_bias_1 = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))

        self.fft_channel_weight_2 = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))
        self.fft_channel_bias_2 = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))

        self.FFC = FourierUnit(hidden_features)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def forward(self, x):
        xinital = self.project_in(x)
        x1,x2 = xinital.chunk(2, dim=1)

        x1 = self.dwconv_1(x1)

        x1 = torch.fft.rfft2(x1)
        x1 = self.fft_channel_weight_1 * x1 + self.fft_channel_bias_1
        x1= torch.fft.irfft2(x1)


        x2 = self.dwconv_2(x2)

        x2 = torch.fft.rfft2(x2)
        x2 = self.fft_channel_weight_2 * x2 + self.fft_channel_bias_2
        x2 = torch.fft.irfft2(x2)


        x = F.silu(x1) * x2
        x = self.project_out(x)
        return x


##=====================================================================================================================#
# multi-head attention Block(MSAttention)

class MHAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MHAttention, self).__init__()

        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习的温度参数，用于控制softmax前的缩放程度，防止梯度消失或注意力过于尖锐
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.softmax = nn.Softmax(dim=-1)
        self.gelu = nn.GELU( )

    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        b, c, h, w = q.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b, head, c, c
        attn = attn.softmax(dim=-1)   # b,head, c, c

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self. project_out(out)

        return out


##########################################################################
# dual domain cross transformer

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads,ffn_expansion_factor, bias):
        super(CrossAttention, self).__init__()

        self.linear_0 = nn.Conv2d(dim, dim*2, 1, 1, 0)

        self.attn  = MHAttention(dim, num_heads,bias)

        self.freblock = MFrequencyBlock(dim,ffn_expansion_factor)

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1)
        )
        self.conv_final = nn.Conv2d(2*dim, dim, kernel_size=1,bias=bias)

    def forward(self, x):

         y, x0 = self.linear_0(x).chunk(2, dim=1)   # x for attention, y for fourierblock

         out_attn = self.attn(x0)

         out_freq = self.freblock(y)
    # --------------------------------------------------------------------------------------------------#
         #第一种方案交叉
         # out_attn = self.channel_interaction(out_attn)
         # out_freq = self.spatial_interaction(out_freq)
         #
         # # cross enhanced learning
         # attened_x = out_attn * torch.sigmoid(out_freq)
         #
         # conv_x = out_freq * torch.sigmoid(out_attn)
    #--------------------------------------------------------------------------------------------------#
        ##第二种方案交叉
         channel_attn = self.channel_interaction(out_attn)
         spatial_freq = self.spatial_interaction(out_freq)

         # cross enhanced learning
         attened_x = out_attn * torch.sigmoid(spatial_freq)

         conv_x = out_freq * torch.sigmoid(channel_attn)

    # --------------------------------------------------------------------------------------------------#
         x_cat = torch.cat([attened_x, conv_x], dim=1)

         x_out = self.conv_final(x_cat)

         return x_out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads,ffn_expansion_factor, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, padding='same', bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- D2CETNet-----------------------
class D2CETNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 # dim=48,
                 dim=32,

                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=1,

                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):

        super(D2CETNet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)    # 初始特征提取层

        self.encoder_level1 = nn.Sequential(*[
                     TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, num_heads=heads[0], bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])   # 第一编码层

        self.down1_2 = Downsample(dim)   ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
                     TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,num_heads=heads[1], bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])   # 第二编码层

        self.down2_3 = Downsample(int(dim * 2 ** 1)) ## from level 2 to level 3
        self.encoder_level3= nn.Sequential(*[
                     TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, num_heads=heads[2], bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])   # 第三编码层

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
                      TransformerBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor, num_heads=heads[3], bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])  # 中间层

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## from level 4 to level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
                     TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, num_heads=heads[2], bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])  # 第三解码层

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
                     TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, num_heads=heads[1],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])  # 第二解码层

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
                     TransformerBlock(dim=int(dim * 2 ** 1),  ffn_expansion_factor=ffn_expansion_factor, num_heads=heads[0],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
                     TransformerBlock(dim=int(dim * 2 ** 1),  ffn_expansion_factor=ffn_expansion_factor,num_heads=heads[0],
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return [out_dec_level1]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    model = D2CETNet().to('cuda')
    macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f"Total FLOPs: {macs}")
    print(f"Total params: {params}")

    print("Params:", count_parameters(model) / 1e6)

    input = torch.randn(1, 3, 64, 64).to('cuda')
    out=model(input)
    print(len(out))

    # from thop import profile
    # from thop import clever_format
    #
    # input = torch.randn(1, 3, 64, 64).to('cuda')
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)