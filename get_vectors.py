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
    for batch in dataset_iter:
        x_array = convert.concat_examples(batch, args.gpu) / 255.
        x_array = F.resize_images(x_array, (64, 64))
        y_array = enc(x_array).data
        if args.gpu >= 0:
            y_array = chainer.cuda.to_cpu(y_array)

        vec_list.append(y_array)

    vector = np.concatenate(vec_list, axis=0)
    vector = vector.mean(axis=0)
    return vector


def main():
    attr_columns = [
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

    parser = argparse.ArgumentParser(description='Get Attribute Vector')
    parser.add_argument('--batchsize', '-b', type=int, default=512,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='data/celebA/',
                        help='Directory of image files.')
    parser.add_argument('--attr_list', '-a', default='data/list_attr_celeba.txt')
    parser.add_argument('--get_attr', default='all', nargs='+', choices=attr_columns + ['all'])
    parser.add_argument('--outfile', '-o', default='attr_vec.json')
    parser.add_argument('--enc', default='pre-trained/enc_iter_310000.npz')
    args = parser.parse_args()

    enc = Encoder()
    npz.load_npz(args.enc, enc)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        enc.to_gpu()

    all_files = os.listdir(args.dataset)
    image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]

    vectors = {}
    attr_df = pd.read_csv(args.attr_list, delim_whitespace=True, header=1)
    if args.get_attr == 'all':
        args.get_attr = attr_columns
    for attr_name in tqdm(list(set(args.get_attr) & set(attr_df.columns))):
        with_attr_files = attr_df[attr_df[attr_name] == 1].index.tolist()
        with_attr_files = list(set(with_attr_files) & set(image_files))
        with_attr_vec = get_vector(enc, with_attr_files, args)

        without_attr_files = attr_df[attr_df[attr_name] != 1].index.tolist()
        without_attr_files = list(set(without_attr_files) & set(image_files))
        without_attr_vec = get_vector(enc, without_attr_files, args)

        vectors[attr_name] = (with_attr_vec - without_attr_vec).tolist()

    with open(args.outfile, 'w') as f:
        f.write(json.dumps(vectors, indent=4, sort_keys=True, separators=(',', ': ')))


if __name__ == '__main__':
    main()
