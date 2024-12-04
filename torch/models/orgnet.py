import torch
import torch.nn as nn


def make_norm(channels, norm="bn"):
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    elif norm == "in":
        return nn.InstanceNorm3d(channels)
    elif norm == "None" or norm is None:
        return nn.Identity()
    else:
        raise ValueError("Implemented normalization layers: bn, in, None")


def make_active(active="relu"):
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leakyrelu": nn.LeakyReLU(inplace=True),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "logsigmoid": nn.LogSigmoid(),
        "sigmoid": nn.Sigmoid(),
        "None": nn.Identity(),
    }
    return activations.get(active, nn.Identity())


class ConvNormActive(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        norm="bn",
        active="relu",
    ):
        super(ConvNormActive, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = make_norm(out_channels, norm)
        self.active = make_active(active)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.active(x)
        return x


class dCNNetEncoder(nn.Module):
    def __init__(self, in_channels, grid_size, norm, active, encoder_config):
        super(dCNNetEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.dim_out = grid_size

        assert len(encoder_config) >= 3

        for out_channels in encoder_config[:-1]:
            kernel_size = 3
            self.encoder.append(
                ConvNormActive(
                    in_channels, out_channels, kernel_size, norm=norm, active=active
                )
            )
            self.dim_out = self._calculate_output_dim(
                self.dim_out, kernel_size, stride=1, padding=1
            )

            self.encoder.append(nn.MaxPool3d(kernel_size, stride=2, padding=1))
            self.dim_out = self._calculate_output_dim(
                self.dim_out, kernel_size, stride=2, padding=1
            )

            in_channels = out_channels

        kernel_size = 2
        self.encoder.append(
            ConvNormActive(
                out_channels,
                encoder_config[-1],
                kernel_size=kernel_size,
                norm=norm,
                active=active,
                padding=0,
            )
        )
        self.output_dim = self._calculate_output_dim(
            self.dim_out, kernel_size, stride=1, padding=0
        )

        self.out_channels = encoder_config[-1]

    def _calculate_output_dim(self, dim, kernel_size, stride, padding, dilation=1):
        return int((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class RegressionHead(nn.Module):
    def __init__(self, enc_out_size, head_act_fn, head_dropout_prob, head_config):
        super(RegressionHead, self).__init__()
        layers = [nn.Flatten()]

        for in_features, out_features in zip(
            [enc_out_size] + head_config[:-1], head_config
        ):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(make_active(head_act_fn))
            layers.append(nn.Dropout(head_dropout_prob))

        layers.append(nn.Linear(head_config[-1], 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class OrgNet(nn.Module):
    def __init__(
        self,
        norm="None",
        encoder_act_fn="relu",
        head_dropout_prob=0.0,
        head_act_fn="gelu",
        n_types=14,
        grid_size=16,
        encoder_config=[16, 80, 400, 512],
        head_config=[512, 128],
    ):
        super(OrgNet, self).__init__()
        self.encoder = dCNNetEncoder(
            n_types, grid_size, norm, encoder_act_fn, encoder_config
        )

        self.head = RegressionHead(
            self.encoder.out_channels * self.encoder.output_dim**3,
            head_act_fn,
            head_dropout_prob,
            head_config,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
