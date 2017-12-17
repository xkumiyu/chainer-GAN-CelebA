import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class Generator(chainer.Chain):

    def __init__(self, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = 100

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, 4 * 4 * 512, initialW=w)
            self.dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=w)
            self.bn0 = L.BatchNormalization(4 * 4 * 512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))), (len(z), 512, 4, 4))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))
        return x


class Discriminator(chainer.Chain):

    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, 64, 3, stride=2, pad=1, initialW=w)
            self.c0_1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w)
            self.c1_0 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=w)
            self.c1_1 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w)
            self.c2_0 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=w)
            self.c2_1 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w)
            self.c3_0 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.l4 = L.Linear(4 * 4 * 512, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(512, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(512, use_gamma=False)

    def __call__(self, x):
        h = add_noise(x)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        h = self.l4(h)
        return h
