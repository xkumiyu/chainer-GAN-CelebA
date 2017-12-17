import argparse
import os

import chainer
from chainer.serializers import npz
from chainer import Variable
import numpy as np
from PIL import Image

from net import Generator


def main():
    parser = argparse.ArgumentParser(description='Generate Image')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--outfile', '-o', default='image.png')
    parser.add_argument('--gen', default='gen.npz', required=True)
    parser.add_argument('--z', '-z', type=float, nargs='+')
    parser.add_argument('--attr_vec', default=None)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    gen = Generator()
    npz.load_npz(args.gen, gen)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()

    if args.z is None:
        z = gen.make_hidden(1)
        if args.verbose:
            print('z:\n{}'.format(' '.join(map(str, z.reshape(-1).tolist()))))
    else:
        if len(args.z) != 100:
            raise ValueError('# of args.z is not 100')
        z = np.asarray(args.z, dtype=np.float32).reshape(1, 100, 1, 1)

    xp = gen.xp
    z = Variable(xp.asarray(z)) / 255.
    x = gen(z)
    x = chainer.cuda.to_cpu(x.data)

    x = x[0]
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, H, W = x.shape
    x = x.transpose(1, 2, 0)

    print('Generate Image and Save to {}'.format(args.outfile))
    Image.fromarray(x).save(args.outfile)



if __name__ == '__main__':
    main()
