import chainer
import chainer.functions as F
from chainer import Variable


class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.k = 0
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        x_real = F.resize_images(x_real, (64, 64))
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        if self.k == 0:
            dis.cache_discriminator_weights()
        if self.k == dis.unrolling_steps:
            gen_optimizer.update(self.loss_gen, gen, y_fake)
            dis.restore_discriminator_weights()
            self.k = -1
        self.k += 1


class EncUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.enc = kwargs.pop('models')
        super(EncUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_real, x_fake):
        loss = F.mean_squared_error(x_real, x_fake)
        chainer.report({'loss': loss}, enc)
        return loss

    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        x_real = F.resize_images(x_real, (64, 64))

        gen, enc = self.gen, self.enc
        z = enc(x_real)
        x_fake = gen(z)

        enc_optimizer.update(self.loss_enc, enc, x_real, x_fake)
