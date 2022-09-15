import torch
import torch.nn as nn


def conv_trans_block_3d(in_dim, out_dim, activation, batch_norm=True, group_norm=0):
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_dim),
            activation(),
        )
    elif group_norm > 0:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(group_norm, out_dim),
            activation(),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation(),
        )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation, batch_norm: bool = True, group_norm=0):
    if batch_norm:
        return nn.Sequential(
            # conv_block_3d(in_dim, out_dim, activation),
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
        )
    elif group_norm > 0:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_norm, out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_norm, out_dim),
        )
    else:
        return nn.Sequential(
            # conv_block_3d(in_dim, out_dim, activation),
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters, batch_norm: bool = True, group_norm=0):
        super(UNet, self).__init__()

        self.in_dim = in_channels
        self.out_dim = n_classes
        self.num_filters = n_filters
        # activation = nn.LeakyReLU(0.2, inplace=True)
        activation = nn.ReLU
        self.batch_norm = batch_norm

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation, self.batch_norm, group_norm)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation, self.batch_norm, group_norm)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation, self.batch_norm, group_norm)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 8, activation, self.batch_norm, group_norm)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.up_1 = conv_block_2_3d(self.num_filters * 16, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.up_2 = conv_block_2_3d(self.num_filters * 16, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, self.batch_norm, group_norm)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation, self.batch_norm, group_norm)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation, self.batch_norm, group_norm)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation, self.batch_norm, group_norm)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation, self.batch_norm, group_norm)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation, self.batch_norm, group_norm)

        # Output
        self.out = nn.Conv3d(self.num_filters, self.out_dim, kernel_size=1)

    def forward(self, x):
        # x -> [None, 1, 64, 128, 128]
        # Down sampling
        down_1 = self.down_1(x)       # -> [None, 16, 64, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [None, 16, 32, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [None, 32, 32, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [None, 32, 16, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [None, 64, 16, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [None, 64, 8, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [None, 128, 8, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [None, 128, 4, 8, 8]

        down_5 = self.down_5(pool_4)  # -> [None, 256, 4, 8, 8]
        pool_5 = self.pool_5(down_5)  # -> [None, 256, 2, 4, 4]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [None, 512, 2, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [None, 512, 4, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out
