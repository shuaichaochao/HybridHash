""" Nested Transformer (NesT) in PyTorch
A PyTorch implement of Aggregating Nested Transformers as described in:
'Aggregating Nested Transformers'
    - https://arxiv.org/abs/2105.12723
The official Jax code is released and available at https://github.com/google-research/nested-transformer. The weights
have been converted with convert/convert_nest_flax.py
Acknowledgments:
* The paper authors for sharing their research, code, and model weights
* Ross Wightman's existing code off which I based this
Copyright 2021 Alexander Soare

Pre-training model parameters(weights from official Google JAX impl)
    'jx_nest_base': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/jx_nest_base-8bc41011.pth'),
"""

import collections.abc
import logging
import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from utils.fx_features import register_notrace_function
from timm.models.layers import helpers
from timm.models.layers import PatchEmbed,Mlp, DropPath, create_classifier, trunc_normal_
from timm.models.layers import create_conv2d, create_pool2d, to_ntuple
from timm.models.registry import register_model
from typing import Callable
_logger = logging.getLogger(__name__)


def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

# 最后一个交互模块放在了hash层里，稍微有些变动
# The last interaction module is placed in the hash layer with slight changes
class HashLayer(nn.Module):
    def __init__(self, config, hash_bit):
        super(HashLayer, self).__init__()

        # self.convS1 = nn.Conv2d(config, config, (1, 3), padding=(0, 1))
        # self.convS2 = nn.Conv2d(config, config, (3, 1), padding=(1, 0))
        #
        # # 不同的池化核大小可得到不同数量的BTs
        # # Different pooling kernel sizes yield different numbers of BTs
        # # self.pool_glo = create_pool2d('avg', kernel_size=7, stride=7, padding=0)
        # # self.global_attention = Interactive_Attention(config, num_heads=8, seq_length=4)
        # # self.upsample = nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False)
        # self.pool_glo = create_pool2d('avg', kernel_size=14, stride=14, padding=0)
        # self.global_attention = Interactive_Attention(config, num_heads=8, seq_length=1)
        # self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=(1,1))
        self.conv2 = nn.Conv2d(config, hash_bit, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.act = nn.Tanh()

    def forward(self, feature):
        # local interaction
        # hash_re = self.convS1(feature)
        # hash_re = self.convS2(hash_re)
        #
        # Global interaction
        # hash_global = self.pool_glo(feature)
        # hash_global = self.global_attention(hash_global)
        # hash_global = hash_global.permute(0, 3, 1, 2)
        # hash_global = self.upsample(hash_global)
        #
        # fusion
        # hash_re = hash_global+hash_re


        # no interaction
        hash_re = self.maxpool(feature)
        hash_re = self.conv2(hash_re)
        hash_re = self.avgpool(hash_re)
        hash_re = torch.flatten(hash_re, start_dim=1, end_dim=3)
        hash_re = self.act(hash_re)

        return hash_re


class Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """
        B, T, N, C = x.shape
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, T, N, C'), permute -> (B, T, N, C', H)
        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, T, N, C)


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Interactive_Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """

    def __init__(self, dim, num_heads=8, seq_length=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_embed_glo = nn.Parameter(torch.zeros(1, seq_length, dim))

    def forward(self, x):

        B, H, W, C = x.permute(0, 2, 3, 1).shape
        x = x.reshape(B,H*W,C)
        # 添加位置信息
        x = x + self.pos_embed_glo
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, N, C'), permute -> (B, N, C', H)
        x = (attn @ v).permute(0, 2, 3, 1).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, H, W, C)


class Interactive_Module(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, pad_type='', num_blocks=0):
        super().__init__()
        self.conv1 = create_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=pad_type, bias=True)
        # self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=pad_type, bias=True)

        # self.conv_glo = create_conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        # # self.pool_glo = create_pool2d('avg', kernel_size=7, stride=7, padding=0)
        # # self.global_attention = Interactive_Attention(out_channels, num_heads=8, seq_length=num_blocks*16)
        # # self.upsample = nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False)
        # self.pool_glo = create_pool2d('avg', kernel_size=14, stride=14, padding=0)
        # self.global_attention = Interactive_Attention(out_channels, num_heads=8, seq_length=num_blocks * 4)
        # self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)

        self.norm = norm_layer(out_channels)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        # Global interaction
        # x_global = self.conv_glo(x)
        # x_global = self.pool_glo(x_global)
        # x_global = self.global_attention(x_global)
        # x_global = x_global.permute(0, 3, 1, 2)
        # x_global = self.upsample(x_global)
        # #
        # local interaction
        # x_min = self.conv(x)
        # #
        # # # fusion
        # x = x_min+x_global
        # x = x_min
        # w_channel = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # w_channel = self.pool(w_channel)

        # no interaction
        x = self.conv1(x)
        w_channel = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        w_channel = self.pool(w_channel)

        return w_channel  # (B, C, H//2, W//2)


def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, H, W, C)
        block_size (int): edge length of a single square block in units of H, W
    """
    B, H, W, C = x.shape
    grid_height = H // block_size
    grid_width = W // block_size
    x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    return x  # (B, T, N, C)


@register_notrace_function  # reason: int receives Proxy
def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired H, W
    """
    B, T, _, C = x.shape
    grid_size = int(math.sqrt(T))
    height = width = grid_size * block_size
    x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x = x.transpose(2, 3).reshape(B, height, width, C)
    return x  # (B, H, W, C)


class HybridHashLevel(nn.Module):

    def __init__(
            self, num_blocks, block_size, seq_length, num_heads, depth, embed_dim, prev_embed_dim=None,
            mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rates=[],
            norm_layer=None, act_layer=None, pad_type=''):
        super().__init__()
        self.block_size = block_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:
            self.pool = Interactive_Module(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type, num_blocks=num_blocks)
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.transformer_encoder = nn.Sequential(*[
            TransformerLayer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rates[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, C, H, W)
        """
        x = self.pool(x)

        x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = x + self.pos_embed
        x = self.transformer_encoder(x)  # (B, T, N, C')
        x = deblockify(x, self.block_size)  # (B, H', W', C')
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 3, 1, 2)  # (B, C, H', W')


class HybridHash(nn.Module):

    def __init__(self, config, num_levels=3, embed_dims=(128, 256, 512),num_heads=(4, 8, 16), depths=(2, 2, 20),
                 mlp_ratio=4., qkv_bias=True,drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5, norm_layer=None,
                 act_layer=None,pad_type='', weight_init='', global_pool='avg'):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            patch_size (int): patch size
            num_levels (int): number of block hierarchies (T_d in the paper)
            embed_dims (int, tuple): embedding dimensions of each level
            num_heads (int, tuple): number of attention heads for each level
            depths (int, tuple): number of transformer layers for each level
            num_classes (int): number of classes for classification head
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim for MLP of transformer layers
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer for transformer layers
            act_layer: (nn.Module): activation layer in MLP of transformer layers
            pad_type: str: Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init: (str): weight init scheme
            global_pool: (str): type of pooling operation to apply to final feature map
        Notes:
            - Default values follow NesT-B from the original Jax code.
            - `embed_dims`, `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - For those following the paper, Table A1 may have errors!
                - https://github.com/google-research/nested-transformer/issues/2
        """
        super().__init__()

        for param_name in ['embed_dims', 'num_heads', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f'Require `len({param_name}) == num_levels`'

        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = config["n_class"]
        self.num_features = embed_dims[-1]
        self.feature_info = []
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels

        self.patch_size = config["patch_size"]

        # Number of blocks at each level
        self.num_blocks = (4 ** torch.arange(num_levels)).flip(0).tolist()
        assert (config["img_size"] // config["patch_size"]) % math.sqrt(self.num_blocks[0]) == 0, \
            'First level blocks don\'t fit evenly. Check `img_size`, `patch_size`, and `num_levels`'

        # Block edge size in units of patches
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(self.num_blocks[0]) is the
        #  number of blocks along edge of image
        self.block_size = int((config["img_size"] // config["patch_size"]) // math.sqrt(self.num_blocks[0]))

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config["img_size"], patch_size=config["patch_size"], in_chans=config["in_chans"], embed_dim=embed_dims[0], flatten=False)
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]

        # Build up each hierarchical level
        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(HybridHashLevel(
                self.num_blocks[i], self.block_size, self.seq_length, num_heads[i], depths[i], dim, prev_dim,
                mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dp_rates[i], norm_layer, act_layer, pad_type=pad_type))
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f'levels.{i}')]
            prev_dim = dim
            curr_stride *= 2

        self.levels = nn.Sequential(*levels)
        # # # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        self.hashlayer_train = HashLayer(self.num_features, config["bit"])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.FcLayer = nn.Linear(embed_dims[-1], config["bit"])
        self.act = nn.Tanh()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        for level in self.levels:
            # 此处修改了
            if hasattr(level, 'pos_embed'):
                trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    def forward_features(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.patch_embed(x)
        x = self.levels(x)
        x_2 = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x_2

    def forward(self, x):
        """ x shape (B, C, H, W)
        """

        x = self.forward_features(x)
        x = self.hashlayer_train(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1, end_dim=3)
        # x = self.FcLayer(x)
        # x = self.act(x)

        return x


def _init_nest_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new):
    """
    Rescale the grid of position embeddings when loading from state_dict
    Expected shape of position embeddings is (1, T, N, C), and considers only square images
    """
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    seq_length_old = posemb.shape[2]
    num_blocks_new, seq_length_new = posemb_new.shape[1:3]
    size_new = int(math.sqrt(num_blocks_new * seq_length_new))
    # First change to (1, C, H, W)
    posemb = deblockify(posemb, int(math.sqrt(seq_length_old))).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=[size_new, size_new], mode='bicubic', align_corners=False)
    # Now change to new (1, T, N, C)
    posemb = blockify(posemb.permute(0, 2, 3, 1), int(math.sqrt(seq_length_new)))
    return posemb

