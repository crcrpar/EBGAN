#!usr/bin/env python
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

"""
h0 = (h + 2p - k) / s + 1
w0 = (w + 2p - k) / s + 1
"""

"""
h0 = s(h-1) + k - 2p
w0 = s(w-1) + k - 2p
"""

class Generator(chainer.Chain):

    def __init__(self, z_dim=50, kernel_size=4, stride=2):
        super(Generator, self).__init__(
            fc1 = L.Linear(z_dim, 1024),
            norm1 = L.BatchNormalization(),
            fc2 = L.Linear(1024, 7*7*128),
            norm2 = L.BatchNormalization(),
            g3 = L.Deconvolution2D(in_channel=1, out_channel=64, ksize=3, stride=stride, pad=1), # 13
            norm3 = L.BatchNormalization(64),
            g4 = L.Deconvolution2D(in_channel=64, out_channel=1, ksize=kernel_size, stride=stride), # 28
        )
        self.z_dim = z_dim

    def __call__(self):
        z = np.random.uniform(size=(self.z_dim))
        h1 = F.relu(self.norm1(self.fc1(z)))
        h2_ = F.relu(self.norm2(self.fc2(h1)))
        h2 = F.reshape(h2_, (-1, 128, 7, 7))
        h3 = F.relu(self.norm3(self.g3(h2)))

        return F.sigmoid(self.g4(h3))


class Discriminator(chainer.Chain):
    # TODO: re-define network.
    # NOTE: 2016/10/20
    
    def __init__(self, in_channels=1, kernel_size=4, stride=2):
        super(Discriminator, self).__init__(
            d1 = L.Convolution2D(in_channel=in_channels, out_channel=64, ksize=kernel_size, stride=stride, pad=1), # 14
            norm1 = L.BatchNormalization(64),
            d2 = L.Convolution2D(in_channel=64, out_channel=128, ksize=kernel_size, stride=stride, pad=1), # 7
            norm2 = L.BatchNormalization(128),
            d3 = L.Convolution2D(in_channel=128, out_channel=64, ksize=kernel_size, stride=stride, pad=1),
            norm3 = L.BatchNormalization(64),
            fc4 = L.Linear(None, 1),
        )

    def __call__(self, x):
        h1 = F.leaky_relu(self.norm1(self.d1(x)))
        h2 = F.leaky_relu(self.norm2(self.d2(h1)))
        h3 = F.leaky_relu(self.norm3(self.d3(h2)))

        return self.fc4(h3)

    def encode(self, x):
        h1 = F.laky_relu(self.norm1(self.d1(x)))
