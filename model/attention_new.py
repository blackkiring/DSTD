import os

from tkinter import CENTER
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from transformer import TransformerClassifier
from tokenizer import Tokenizer

class ViTLite(nn.Module):
    def __init__(self,
                 img_size=16,
                 embedding_dim=128,
                 x_input_channels=4,
                 y_input_channels=4,
                 kernel_size=3,
                 dropout=0.2,
                 attention_dropout=0.2,
                 stochastic_depth=0.1,
                 num_layers=3,
                 num_heads=2,
                 mlp_ratio=1,
                 num_classes=7,
                 positional_embedding=None,
                 *args, **kwargs):
        super(ViTLite, self).__init__()
#        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
#                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer1 = Tokenizer(n_input_channels=x_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride= max(1, (kernel_size // 2) - 1),
                                   padding=max(1, (kernel_size // 2)),
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=2,
                                   conv_bias=False)
        self.tokenizer2=Tokenizer(n_input_channels=y_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride= max(1, (kernel_size // 2) - 1),
                                   padding=max(1, (kernel_size // 2)),
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=2,
                                   conv_bias=False)
        self.classifier = TransformerClassifier(
            sequence_length1=self.tokenizer1.sequence_length(n_channels=x_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            sequence_length2=self.tokenizer2.sequence_length(n_channels=y_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            image_size=img_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x,y):
        x = self.tokenizer1(x)
        y=  self.tokenizer2(y)
        return self.classifier(x,y)

