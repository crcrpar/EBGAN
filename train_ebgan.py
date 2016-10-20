#!usr/bin/env python

from __future__ import print_function
import os
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.datasets import get_mnist
from chainer.training import trainer, extension
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module

import net

def pt_regularizer(S, bs=None):
    """
    Args:
        S (chainer.Variable): contain ndarray, whose shape is (batch_size, latent_dim)
    Returns:
        pt (chainer.Variable): scalar
    """
    if bs is None:
        bs = S.shape[0]
    S2 = F.batch_l2_norm_squared(S) #
    S2 = F.reshape(S2, shape=(S2.data.shape[0], 1))
    S2 = F.matmul(S2, S2, transa=False, transb=True)

    dotS = F.matmul(S, S, transa=False, transb=True)
    dotS = dotS * dotS

    assert(S2.data.shape == dotS.data.shape)

    pt =  (F.sum(dotS / S2) - bs) / float(bs*(bs-1))

    return pt

class EBGAN_Updater(chainer.training.StandardUpdater):

    def __init__(self, iterator, generator, discriminator, optimizers, converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}

        self._iterators = iterator
        self._optimizers = optimizers

        self.gen = generator
        self.dis = discriminator

        self.converter = converter
        self.iteration = 0
        self.device = device

    def updater_core(self, x):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        fake_image = self.gan()
        _data = F.concat((chainer.Variable(in_arrays), fake_image))
        prediction = self.dis(_data)

        mse_dis = F.mean_squared_error(_data, prediction)
