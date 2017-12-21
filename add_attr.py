import argparse
import json
import os

import chainer
import chainer.functions as F
from chainer.serializers import npz
from chainer import Variable
import numpy as np
from PIL import Image

from dataset import CelebADataset
from net import Encoder
from net import Generator


def array_to_image(x):
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    x = x.transpose(1, 2, 0)
    return x


def get_add_vec(attr_vec, attr_rate):
    add_vec = np.zeros(100, dtype=np.float32)
    for attr_name, attr_val in attr_vec.items():
        add_vec += np.asarray(attr_vec[attr_name], dtype=np.float32) * attr_rate[attr_name]
    return add_vec


def encode(args):
    enc = Encoder()
    npz.load_npz(args.enc, enc)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        enc.to_gpu()
    xp = enc.xp

    image = CelebADataset([args.infile])[0]
    x = Variable(xp.asarray([image])) / 255.
    x = F.resize_images(x, (64, 64))

    with chainer.using_config('train', False):
        z = enc(x)
    return z, x.data[0]


def generate(z, args):
    gen = Generator()
    npz.load_npz(args.gen, gen)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()

    if z is None:
        z = gen.make_hidden(1)

    with chainer.using_config('train', False):
        x = gen(z)
    x = x.data[0]
    return x


def main():
    attrs = [
        '5_o_Clock_Shadow',
        'Arched_Eyebrows',
        'Attractive',
        'Bags_Under_Eyes',
        'Bald',
        'Bangs',
        'Big_Lips',
        'Big_Nose',
        'Black_Hair',
        'Blond_Hair',
        'Blurry',
        'Brown_Hair',
        'Bushy_Eyebrows',
        'Chubby',
        'Double_Chin',
        'Eyeglasses',
        'Goatee',
        'Gray_Hair',
        'Heavy_Makeup',
        'High_Cheekbones',
        'Male',
        'Mouth_Slightly_Open',
        'Mustache',
        'Narrow_Eyes',
        'No_Beard',
        'Oval_Face',
        'Pale_Skin',
        'Pointy_Nose',
        'Receding_Hairline',
        'Rosy_Cheeks',
        'Sideburns',
        'Smiling',
        'Straight_Hair',
        'Wavy_Hair',
        'Wearing_Earrings',
        'Wearing_Hat',
        'Wearing_Lipstick',
        'Wearing_Necklace',
        'Wearing_Necktie',
        'Young']

    parser = argparse.ArgumentParser(description='Generate Image')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--infile', '-i', default='data/celebA/000001.jpg')
    parser.add_argument('--outdir', '-o', default='.')
    parser.add_argument('--enc', default='pre-trained/enc_iter_310000.npz')
    parser.add_argument('--gen', default='pre-trained/gen_iter_310000.npz')
    parser.add_argument('--attr_vec', default='pre-trained/attr_vec_Young.json')
    for attr in attrs:
        parser.add_argument('--' + attr, default=0, type=int)
    args = parser.parse_args()

    with open(args.attr_vec, 'r') as f:
        attr_vec = json.load(f)
    base, ext = os.path.splitext(os.path.basename(args.infile))

    if args.infile is None:
        x1 = generate(None, args)
    else:
        z, x0 = encode(args)
        x1 = generate(z, args)
        Image.fromarray(array_to_image(x0)).save(
            os.path.join(args.outdir, '{}_0{}'.format(base, ext)))

    Image.fromarray(array_to_image(x1)).save(
        os.path.join(args.outdir, '{}_1{}'.format(base, ext)))

    v = get_add_vec(attr_vec, vars(args))
    if v.mean() != 0:
        x2 = generate(z + v, args)
        Image.fromarray(array_to_image(x2)).save(
            os.path.join(args.outdir, '{}_2{}'.format(base, ext)))

    # v = get_add_vec(attr_vec, {'Young': 1})
    # x2 = generate(z + v, args)
    # Image.fromarray(array_to_image(x2)).save(
    #     os.path.join(args.outdir, '{}_Young+1{}'.format(base, ext)))
    #
    # v = get_add_vec(attr_vec, {'Young': 2})
    # x2 = generate(z + v, args)
    # Image.fromarray(array_to_image(x2)).save(
    #     os.path.join(args.outdir, '{}_Young+2{}'.format(base, ext)))
    #
    # v = get_add_vec(attr_vec, {'Young': -1})
    # x2 = generate(z + v, args)
    # Image.fromarray(array_to_image(x2)).save(
    #     os.path.join(args.outdir, '{}_Young-1{}'.format(base, ext)))


if __name__ == '__main__':
    main()
