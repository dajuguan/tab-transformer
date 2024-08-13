#!/usr/bin/env python3

from __future__ import annotations
import typing
import torch
import sys
sys.path.append("..")
from tab_transformer_pytorch import TabTransformer, FTTransformer
from tab_transformer_pytorch.ft_transformer import NumericalEmbedder

from torch import nn
from einops import rearrange

TensorFunc = typing.Callable[[torch.Tensor], torch.Tensor]

__all__ = ('FCNN', 'HFNN', 'MFNN')


class BasicBlock(torch.nn.Module):
    """Basic block for a fully-connected layer."""

    def __init__(self, in_features: int,
                 out_features: int,
                 activation: type[torch.nn.Module]):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features)
        self.activation = activation()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.activation(x)
        return x


class FCNN(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 midlayer_features: list[int],
                 dim_rope_seq: int = 0,
                 activation: type[torch.nn.Module] = torch.nn.Tanh,
                 low_fidelity_features: int = 0,
                 enable_attention=True,
                 enable_embedding=False
                 ):
        super().__init__()

        attn_features = 16
        embedding_dim = 32
        self.attention = FTTransformer(
            categories = (),      # tuple containing the number of unique values within each category
            num_continuous = in_features + low_fidelity_features,                # number of continuous values
            dim = embedding_dim,                           # dimension, paper set at 32
            dim_rope_seq= dim_rope_seq,
            dim_out = attn_features,             # binary prediction, but could be anything
            depth = 6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.0,                 # post-attention dropout
            ff_dropout = 0.0                   # feed forward dropout
        )

        if enable_attention:
            print("Attention enabled")
            x_low_feature_dim = attn_features + low_fidelity_features
            x_low_feature_dim = attn_features
        elif enable_embedding:
            print("Num embedding enabled")
            embedding_dim = 32
            x_low_feature_dim = embedding_dim * in_features + low_fidelity_features
            self.numerical_embedder = NumericalEmbedder(embedding_dim, in_features)
        else:
            print("Simple MLP")
            x_low_feature_dim = low_fidelity_features + in_features
            self.numerical_embedder = NumericalEmbedder(embedding_dim, self.attention.num_continuous)
        layer_sizes = [x_low_feature_dim] + midlayer_features + [out_features]
        self.layers = torch.nn.Sequential(*[
            BasicBlock(layer_sizes[i], layer_sizes[i+1], activation)
            for i in range(len(layer_sizes) - 1)
        ])
        self.fc = torch.nn.Linear(low_fidelity_features , out_features)
        self.enable_attention = enable_attention
        self.enable_embedding = enable_embedding

    def forward(self, x: torch.Tensor, y_low: torch.Tensor) -> torch.Tensor:
        x_categ = torch.tensor([]) # dummy categ need by ft_transformer
        if self.enable_attention:
            # x = self.attention(x_categ, x)
            x = torch.concat([x, y_low], dim=1)
            x_combine_ylow = self.attention(x_categ, x)
        elif self.enable_embedding:
            # x = torch.concat([x, y_low], dim=1)
            x = self.numerical_embedder(x)
            x = rearrange(x, 'b n h -> b (n h)')
            x_combine_ylow = torch.concat([x, y_low], dim=1)
        else:
            x_combine_ylow = torch.concat([x, y_low], dim=1)
        y_nonlinear = self.layers(x_combine_ylow)  # non-linear term
        y_linear = self.fc(y_low)
        return y_nonlinear

if __name__ == '__main__':

    x = torch.linspace(0, 1, 2).reshape(-1, 1)
    y_low = torch.linspace(0, 1, 2 * 4).reshape(-1, 4)

    lfnn = FCNN(1, 5, [16, 16, 16], activation=torch.nn.Tanh, low_fidelity_features=4)
    print("lfnn", lfnn)
    y_high = lfnn(x, y_low)
    print("y_high", y_high)