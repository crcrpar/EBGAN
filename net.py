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

    def __init__(self, batch_size=20, z_dim=50, kernel_size=4, stride=2):
        super(Generator, self).__init__(
            fc1 = L.Linear(z_dim, 1024),
            norm1 = L.BatchNormalization(1024),
            fc2 = L.Linear(1024, 7*7*128),
            norm2 = L.BatchNormalization(6272),
            g3 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=3, stride=stride, pad=1), # 13
            norm3 = L.BatchNormalization(64),
            g4 = L.Deconvolution2D(in_channels=64, out_channels=1, ksize=kernel_size, stride=stride), # 28
        )
        self.z_dim = z_dim
        self.batch_size = batch_size

    def __call__(self):
        z = np.random.uniform(size=(self.batch_size, self.z_dim)).astype(np.float32)
        #print('z.shape', z.shape)
        h1_ = self.fc1(z)
        #print(h1_.data.shape)
        h1 = F.relu(self.norm1(h1_))
        h2_ = F.relu(self.norm2(self.fc2(h1)))
        h2 = F.reshape(h2_, (-1, 128, 7, 7))
        h3 = F.relu(self.norm3(self.g3(h2)))

        return F.tanh(self.g4(h3))


class Discriminator(chainer.Chain):
    # TODO: re-define network.
    # NOTE: 2016/10/20

    def __init__(self, z_dim = 50, in_channels=1, kernel_size=4, stride=2):
        super(Discriminator, self).__init__(
            d1 = L.Convolution2D(in_channels=in_channels, out_channels=64, ksize=kernel_size, stride=stride, pad=1), # 14
            norm1 = L.BatchNormalization(64),
            d2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=kernel_size, stride=stride, pad=1), # 7
            norm2 = L.BatchNormalization(128),
            d3 = L.Convolution2D(in_channels=128, out_channels=64, ksize=kernel_size, stride=stride, pad=1),
            norm3 = L.BatchNormalization(64),
            fc4 = L.Linear(None, z_dim),
            norm4 = L.BatchNormalization(50),

            fc5 = L.Linear(z_dim, 1024),
            norm5 = L.BatchNormalization(1024),
            fc6 = L.Linear(1024, 7*7*128),
            norm6 = L.BatchNormalization(7*7*128),
            g7 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=3, stride=stride, pad=1),
            norm7 = L.BatchNormalization(64),
            g8 = L.Deconvolution2D(in_channels=64, out_channels=1, ksize=kernel_size, stride=stride),

        )
        self.z_dim = z_dim

    def __call__(self, x):

        return self.generate(self.encode(x))

    def encode(self, x):
        h1 = F.relu(self.norm1(self.d1(x)))
        h2 = F.relu(self.norm2(self.d2(h1)))
        h3 = F.relu(self.norm3(self.d3(h2)))

        return self.fc4(h3)

    def generate(self, z):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.norm5(h5))
        h6 = F.relu(self.fc6(h6))
        h7 = F.relu(self.norm6(h6))
        h7 = F.reshape(h7, (-1, 128, 7, 7))
        h8 = F.relu(self.norm7(self.g7(h7)))
        h9 = self.g8(h8)

        return F.sigmoid(h9)
