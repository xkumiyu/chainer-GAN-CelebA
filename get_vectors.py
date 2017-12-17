import argparse
import json
import os

import chainer
from chainer.dataset import convert
import chainer.functions as F
from chainer.serializers import npz
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import CelebADataset
from net import Encoder


def get_vector(enc, image_files, args):
    # Setup dataset
    dataset = CelebADataset(paths=image_files, root=args.dataset)
    dataset_iter = chainer.iterators.SerialIterator(dataset, args.batchsize,
                                                    repeat=False, shuffle=False)

    # Infer
    vec_list = []
    pbar = tqdm(total=len(dataset) // args.batchsize)
    for batch in dataset_iter:
        x_array = convert.concat_examples(batch, args.gpu) / 255.
        x_array = F.resize_images(x_array, (64, 64))
        y_array = enc(x_array).data
        if args.gpu >= 0:
            y_array = chainer.cuda.to_cpu(y_array)

        vec_list.append(y_array)
        pbar.update()
    pbar.close()

    vector = np.concatenate(vec_list, axis=0)
    vector = vector.mean(axis=0)
    return vector


def main():
    parser = argparse.ArgumentParser(description='Get Attribute Vector')
    parser.add_argument('--batchsize', '-b', type=int, default=512,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='data/celebA/',
                        help='Directory of image files.')
    parser.add_argument('--attr_list', '-a', default='data/list_attr_celeba.txt')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--enc', default='enc.npz', required=True)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# batchsize: {}'.format(args.batchsize))
    print('')

    enc = Encoder()
    npz.load_npz(args.enc, enc)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        enc.to_gpu()

    all_files = os.listdir(args.dataset)
    image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]

    vectors = {}
    attr_df = pd.read_csv(args.attr_list, delim_whitespace=True, header=1)
    for attr_name in attr_df.columns:
        with_attr_files = attr_df[attr_df[attr_name] == 1].index.tolist()
        with_attr_files = list(set(with_attr_files) & set(image_files))
        print('{} image files with {}'.format(len(with_attr_files), attr_name))
        with_attr_vec = get_vector(enc, with_attr_files, args)

        without_attr_files = attr_df[attr_df[attr_name] != 1].index.tolist()
        without_attr_files = list(set(without_attr_files) & set(image_files))
        print('{} image files with {}'.format(len(without_attr_files), attr_name))
        without_attr_vec = get_vector(enc, without_attr_files, args)

        vectors[attr_name] = (with_attr_vec - without_attr_vec).tolist()
        break

    with open(os.path.join(args.out, 'attr_vec.json'), 'w') as f:
        f.write(json.dumps(vectors, indent=4, sort_keys=True, separators=(',', ': ')))


if __name__ == '__main__':
    main()
