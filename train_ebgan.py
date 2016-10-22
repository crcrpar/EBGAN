#!usr/bin/env python

from __future__ import print_function
import argparse
import os
import six
import numpy as np
import datetime

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.datasets import get_mnist
from chainer.training import trainer, extension, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.datasets import get_mnist
from chainer import optimizer as optimizer_module
from chainer import reporter

import net

"""
REF: https://github.com/pfnet/chainer/issues/1786
Unused params cause NoneType error
"""

def pt_regularizer(S, bs=None):
    """
    Args:
        S (chainer.Variable): contain ndarray, whose shape is (batch_size, latent_dim)
        bs (int): batch size.
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

    pt =  (F.sum(dotS / S2) - chainer.Variable(np.array(bs).astype(np.float32))) / float(bs*(bs-1))

    return pt

class EBGAN_Updater(chainer.training.StandardUpdater):

    def __init__(self, iterator, generator, discriminator, optimizers, batch_size, margin=1.0, coeff=0.1, converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self._optimizers = optimizers
        self.gen = generator
        self.dis = discriminator
        self.m = chainer.Variable(np.array(margin).astype(np.float32))
        self.zero = chainer.Variable(np.zeros(shape=(1)).astype(np.float32))
        self._c = coeff
        self.converter = converter
        self.iteration = 0
        self.device=device
        self.batch_size = batch_size

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        fake_image = self.gen()

        reconstructed_true = self.dis(chainer.Variable(in_arrays))
        reconstructed_false = self.dis(fake_image)

        mse_false_rt = F.mean_squared_error(reconstructed_false, fake_image)
        loss_gen = mse_false_rt + self._c * pt_regularizer(self.dis.encode(fake_image), bs=self.batch_size)

        mse_true_rt = F.mean_squared_error(reconstructed_true, chainer.Variable(in_arrays))

        loss_dis_ = self.m - mse_false_rt
        if loss_dis_.data >= .0:
            loss_dis = mse_true_rt + loss_dis_
        else:
            loss_dis = mse_true_rt
        reporter.report({'dis/loss': loss_dis, 'gen/loss': loss_gen})

        loss_dictionary = {'dis':loss_dis, 'gen':loss_gen}
        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.cleargrads()
            loss_dictionary[name].backward()
            optimizer.update()

def main():
    parser = argparse.ArgumentParser(description='Train EBGAN on MNIST.')
    parser.add_argument('--latent_dim', '-l', type=int, default=20, help='dimension of latent space.')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='learning epoch')
    parser.add_argument('--batchsize', '-b', type=int, default=50)
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='negative integer indicates only CPU')
    parser.add_argument('--resume', '-r', type=str, help='trained snapshot')
    parser.add_argument('--out', '-o', type=str, help='directory to save')
    parser.add_argument('--loaderjob', type=int, help='loader job for parallel iterator')
    parser.add_argument('--interval', '-i', default=10, type=int, help='frequency of snapshot. larger integer indicates less snapshots.')

    args = parser.parse_args()
    n_epoch = args.epoch
    latent_dim = args.latent_dim
    batch_size = args.batchsize
    if args.gpu >= 0:
        xp = cuda.cupy

    generator = net.Generator(batch_size=batch_size, z_dim = latent_dim)
    discriminator = net.Discriminator1()
    if args.gpu >= 0:
        generator.to_gpu()
        discriminator.to_gpu()

    opt_gen = chainer.optimizers.Adam()
    opt_dis = chainer.optimizers.Adam()
    opt_gen.setup(generator)
    opt_dis.setup(discriminator)
    optimizers = {'dis': opt_dis, 'gen': opt_gen}

    mnist, val = get_mnist(withlabel=False, ndim=3)
    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(mnist, batch_size=args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(val, batch_size=args.batchsize, n_processes=args.loaderjob)
    else:
        train_iter = chainer.iterators.SerialIterator(mnist, batch_size)
        val_iter = chainer.iterators.SerialIterator(val, batch_size)
    updater = EBGAN_Updater(iterator=train_iter, generator=generator, discriminator=discriminator, optimizers=optimizers, batch_size=batch_size)

    log_name = datetime.datetime.now().strftime('%m_%d_%H_%M') + '_log.json'
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'))
    print('# num epoch: {}\n'.format(n_epoch))
    trainer.extend(extensions.dump_graph('gen/loss', out_name='gen_loss.dot'))
    trainer.extend(extensions.dump_graph('dis/loss', out_name='dis_loss.dot'))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.LogReport(log_name=log_name+'{iteration}'))
    trainer.extend(extensions.PrintReport(['epoch', 'dis/loss', 'gen/loss']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
