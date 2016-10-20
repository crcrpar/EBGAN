#!usr/bin/env python

from __future__ import print_function
import os
import six
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
from chaienr import reporter

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

    pt =  (F.sum(dotS / S2) - bs) / float(bs*(bs-1))

    return pt

class EBGAN_Updater(chainer.training.StandardUpdater):

    def __init__(self, iterator, generator, discriminator, optimizers, coeff=0.1 converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}

        self._iterators = iterator
        self._optimizers = optimizers

        self.gen = generator
        self.dis = discriminator
        self._c = coeff
        self.converter = converter
        self.iteration = 0
        self.device = device

    def updater_core(self, x):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        fake_image = self.gan()

        reconstructed_true = self.dis(chainer.Variable(in_arrays))
        reconstructed_false = self.dis(fake_image)
        loss_gen = F.sqrt(F.sum(F.batch_l2_norm_squared(reconstructed_false - fake_image)))

        loss_dis = F.sqrt(F.sum(F.batch_l2_norm_squared(reconstructed_true - chainer.Variable(in_arrays)))) + F.sqrt(F.sum(F.batch_l2_norm_squared(reconstructed_false - fake_image)))

        reporter.report({'dis/loss': loss_dis, 'gen/loss': loss_gen})

        loss_dictionary = {'dis':loss_dis, 'gen':loss_gen}

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.cleargrads()
            loss_dictionary[name].backward()
            optimizer.update()

            
