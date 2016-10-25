#!usr/bin/env python

from __future__ import print_function
import argparse
import copy
import os
import six
import numpy as np
import datetime
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, training
from chainer.datasets import get_mnist
from chainer.training import trainer, extension, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.datasets import get_mnist
from chainer import optimizer as optimizer_module
from chainer import reporter as reporter_module
from chainer import reporter

import net

"""
REF: https://github.com/pfnet/chainer/issues/1786
Unused params cause NoneType error
"""

def pt_regularizer(S, bs=None, train=True):
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

    if train:
        pt =  (F.sum(dotS / S2) - chainer.Variable(np.array(bs).astype(np.float32))) / float(bs*(bs-1))

    else:
        pt =  (F.sum(dotS / S2) - chainer.Variable(np.array(bs).astype(np.float32), volatile='on')) / float(bs*(bs-1))

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
        self.zero = chainer.Variable(np.array(0.0).astype(np.float32))
        self._c = coeff
        self.converter = converter
        self.iteration = 0
        self.device=device
        self.batch_size = batch_size

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        in_arrays = chainer.Variable(in_arrays)
        fake_image = self.gen()
        dis_input = F.concat((in_arrays, fake_image), axis=0)
        dis_output = self.dis(dis_input)
        (reconstructed_true, reconstructed_false) = F.split_axis(dis_output, 2, axis=0)

        mse_false_rt = F.mean_squared_error(reconstructed_false, fake_image)
        loss_gen = mse_false_rt + self._c * pt_regularizer(self.dis.encode(fake_image), bs=self.batch_size)

        mse_true_rt = F.mean_squared_error(reconstructed_true, in_arrays)

        loss_dis_ = self.m - mse_false_rt
        loss_dis = mse_true_rt + F.maximum(self.zero, loss_dis_)
        '''if loss_dis_.data >= .0:
            loss_dis = mse_true_rt + loss_dis_
        else:
            loss_dis = mse_true_rt'''
        reporter.report({'dis/loss': loss_dis, 'gen/loss': loss_gen})

        loss_dictionary = {'dis':loss_dis, 'gen':loss_gen}
        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.cleargrads()
            loss_dictionary[name].backward()
            optimizer.update()

class EBGAN_Evaluator(chainer.training.extensions.Evaluator):

    trigger=1, 'epoch'
    default_name='validation'
    priority=chainer.training.extension.PRIORITY_WRITER

    def __init__(self, iterator, gen, dis, batchsize, coeff=0.1, margin=1.0,  converter=convert.concat_examples, device=None, eval_hook=None, eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'gen':gen, 'dis':dis}

        self.converter = converter
        self.device = device
        self.eval_hook=eval_hook
        self.eval_func = eval_func
        self.m = chainer.Variable(np.array(margin).astype(np.float32))
        self.zero = chainer.Variable(np.array(0.0).astype(np.float32))
        self._c = coeff

    def get_iterator(self, name):
        return sefl._iterators['name']

    def get_all_iterators(self):
        return dict(self._iterators)

    def get_target(self, name):
        return self._targets[name]

    def get_all_targets(self):
        return dict(self._targets)

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    def evaluate(self):
        iterator = self._iterators['main']
        gen = self._targets['gen']
        dis = self._targets['dis']

        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                batch_size = in_arrays.shape[0]
                fake_image = gen(train=False)

                reconstructed_true = dis(chainer.Variable(in_arrays, volatile='on'))
                reconstructed_false = dis(fake_image)

                mse_false_rt = F.mean_squared_error(reconstructed_false, fake_image)
                loss_gen = mse_false_rt + self._c * pt_regularizer(dis.encode(fake_image), train=False)

                mse_true_rt = F.mean_squared_error(reconstructed_true, chainer.Variable(in_arrays, volatile='on'))

                loss_dis_ = self.m - mse_false_rt
                loss_dis = mse_true_rt + F.maximum(self.zero, loss_dis_)
                '''if loss_dis_.data >= .0:
                    loss_dis = mse_true_rt + loss_dis_
                else:
                    loss_dis = mse_true_rt'''

                observation['dis/val/loss'] = loss_dis
                observation['gen/val/loss'] = loss_gen

                del loss_dis
                del loss_gen

            summary.add(observation)

        return summary.compute_mean()


def main():
    parser = argparse.ArgumentParser(description='Train EBGAN on MNIST.')
    parser.add_argument('--latent_dim', '-l', type=int, default=20, help='dimension of latent space.')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='learning epoch')
    parser.add_argument('--batchsize', '-b', type=int, default=20)
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='negative integer indicates only CPU')
    parser.add_argument('--resume', '-r', type=str, help='trained snapshot')
    parser.add_argument('--out', '-o', default='images/', type=str, help='directory to save images')
    parser.add_argument('--loaderjob', type=int, help='loader job for parallel iterator')
    parser.add_argument('--interval', '-i', default=1, type=int, help='frequency of snapshot. larger integer indicates less snapshots.')
    parser.add_argument('--test', type=int, default=-1, help='positive integer indicates debug mode.')

    args = parser.parse_args()
    n_epoch = args.epoch
    latent_dim = args.latent_dim
    batch_size = args.batchsize
    if args.gpu >= 0:
        xp = cuda.cupy

    if not os.path.exists(args.out):
        os.mkdir(args.out)

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
    if args.test > -1:
        N = mnist.shape[0]
        N = int(N / 100)
        mnist = mnist[:N, :, :, :]
        print('### DEBUG ###\n#dataset size: {}'.format(N))
        N = val.shape[0]
        N = int(N / 100)
        val = val[:N, :,:,:]
        print('#validation set: {}\n'.format(N))

    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x_val_known = chainer.Variable(np.asarray(mnist[train_ind]), volatile='on')
    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x_val = chainer.Variable(np.asarray(val[test_ind]), volatile='on')

    if args.loaderjob:
        train_iter = chainer.iterators.MultiprocessIterator(mnist, batch_size=args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(val, batch_size=args.batchsize, n_processes=args.loaderjob, repeat=False, shuffle=False)
    else:
        train_iter = chainer.iterators.SerialIterator(mnist, batch_size)
        val_iter = chainer.iterators.SerialIterator(val, batch_size, repeat=False, shuffle=False)
        # repeat & shuffle might make learning slower

    updater = EBGAN_Updater(iterator=train_iter, generator=generator, discriminator=discriminator, optimizers=optimizers, batch_size=batch_size, device=args.gpu)

    log_name = datetime.datetime.now().strftime('%H_%M_epoch')
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'))
    print('# num epoch: {}\n'.format(n_epoch))

    if args.test == -1:
        trainer.extend(extensions.dump_graph('gen/loss', out_name='gen_loss.dot'))
        trainer.extend(extensions.dump_graph('dis/loss', out_name='dis_loss.dot'))

    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'))
    trainer.extend(extensions.LogReport(log_name='log_'+'{epoch}'+'.json'))
    trainer.extend(extensions.PrintReport(['epoch', 'dis/loss', 'gen/loss', 'dis/val/loss', 'gen/val/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.extend(EBGAN_Evaluator(val_iter, trainer.updater.gen, trainer.updater.dis, batchsize=batch_size, device=args.gpu))

    @training.make_extension(trigger=(1, 'epoch'))
    def save_image(trainer):

        def save_img(x, filename):
            fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
            for ai, xi in zip(ax.flatten(), x):
                ai.imshow(xi[0])
                ai.colorbar()
            fig.savefig(filename)
            plt.close('all')

        train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
        test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
        # number of inputs are 9
        known_inputs = mnist[train_ind]
        unknown_inputs = val[test_ind]
        known_reconstructed = trainer.updater.dis(chainer.Variable(known_inputs, volatile='on')).data
        unknown_reconstructed = trainer.updater.dis(chainer.Variable(unknown_inputs, volatile='on')).data
        if args.gpu > -1:
            known_reconstructed = cuda.to_cpu(known_reconstructed)
            unknown_reconstructed = cuda.to_cpu(unknown_reconstructed)
        save_img(known_inputs, os.path.join(args.out, 'known_inputs_{}.png'.format(trainer.updater.epoch)))
        save_img(unknown_inputs, os.path.join(args.out, 'unknown_inputs_{}.png'.format(trainer.updater.epoch)))
        save_img(known_reconstructed, os.path.join(args.out, 'known_resconstructed_{}.png'.format(trainer.updater.epoch)))
        save_img(unknown_reconstructed, os.path.join(args.out, 'unknown_reconstructed_{}.png'.format(trainer.updater.epoch)))


    trainer.extend(save_image)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
