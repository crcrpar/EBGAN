#!usr/bin/env python
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

""" Convolution2D
h0 = (h + 2p - k) / s + 1
w0 = (w + 2p - k) / s + 1
"""

""" Deconvolution2D
h0 = s(h-1) + k - 2p
w0 = s(w-1) + k - 2p
"""

class Generator(chainer.Chain):

    def __init__(self, batch_size=20, z_dim=50, kernel_size=4, stride=2):
        super(Generator, self).__init__(
            fc1 = L.Linear(z_dim, 1024, initialW=W, bias=bias),
            norm1 = L.BatchNormalization(1024),
            fc2 = L.Linear(1024, 7*7*128, initialW=W, bias=bias),
            norm2 = L.BatchNormalization(6272),
            g3 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=3, stride=stride, pad=1, initialW=W), # 13
            norm3 = L.BatchNormalization(64),
            g4 = L.Deconvolution2D(in_channels=64, out_channels=1, ksize=kernel_size, stride=stride, initialW=W), # 28
        )
        self.z_dim = z_dim
        self.batch_size = batch_size
        W = initializers.HeNormal(1 / np.sqrt(2), np.float32)
        bias = initializers.Zero(np.float32)


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
            d1 = L.Convolution2D(in_channels=in_channels, out_channels=64, ksize=kernel_size, stride=stride, pad=1, initialW=W), # 14
            norm1 = L.BatchNormalization(64),
            d2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=kernel_size, stride=stride, pad=1, initialW=W), # 7
            norm2 = L.BatchNormalization(128),
            d3 = L.Convolution2D(in_channels=128, out_channels=64, ksize=kernel_size, stride=stride, pad=1, initialW=W),
            norm3 = L.BatchNormalization(64),
            fc4 = L.Linear(None, z_dim),
            norm4 = L.BatchNormalization(50),

            fc5 = L.Linear(z_dim, 13*13*64),
            norm5 = L.BatchNormalization(13*13*64),
            g6 = L.Deconvolution2D(in_channels=64, out_channels=1, ksize=kernel_size, stride=stride),
        )
        self.z_dim = z_dim
        W = initializers.HeNormal(1 / np.sqrt(2), np.float32)
        bias = initializers.Zero(np.float32)

    def __call__(self, x):
        h1 = F.relu(self.norm1(self.d1(x)))
        h2 = F.relu(self.norm2(self.d2(h1)))
        h3 = F.relu(self.norm3(self.d3(h2)))
        h4 = self.fc4(h3)

        h5 = F.relu(self.norm5(self.fc5(h4)))
        h5 = F.reshape(h5, (-1, 64, 13, 13))
        h6 = F.sigmoid(self.g6(h5))

        return h6

    def encode(self, x):
        h1 = F.relu(self.norm1(self.d1(x)))
        h2 = F.relu(self.norm2(self.d2(h1)))
        h3 = F.relu(self.norm3(self.d3(h2)))
        h4 = self.fc4(h3)
        return h4

    def generate(self, h4):
        h5 = F.relu(self.norm5(self.fc5(h4)))
        h5 = F.reshape(h5, (-1, 64, 13, 13))
        h6 = F.sigmoid(self.g6(h5))

        return h6
