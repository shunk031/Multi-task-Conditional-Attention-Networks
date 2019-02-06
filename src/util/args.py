import argparse

from models import ARCHS
from util import const


def common_args():

    parser = argparse.ArgumentParser(description='training for analysing creatives')
    parser.add_argument('--seed',
                        type=int,
                        default=19950815)
    parser.add_argument('--test',
                        action='store_true',
                        default=False)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False)
    parser.add_argument('--out',
                        type=str,
                        default='result')
    parser.add_argument('--arch',
                        type=str,
                        default='gru',
                        choices=ARCHS.keys())
    parser.add_argument('--gpu',
                        type=int,
                        default=-1)
    parser.add_argument('--fold',
                        type=int,
                        default=5)
    parser.add_argument('--group',
                        action='store_true',
                        default=False)
    parser.add_argument('--training_type',
                        type=str,
                        choices=const.TRAINING_TYPES,
                        default=const.TRAINING_TYPES_REGRESSION)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--resume', '-r', default='')

    return parser


def parse_train_args():

    parser = common_args()
    parser.add_argument('--epoch',
                        type=int,
                        default=50)
    parser.add_argument('--batchsize',
                        type=int,
                        default=32)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0001)
    parser.add_argument('--cv',
                        action='store_true',
                        default=False)
    parser.add_argument('--layer',
                        type=int,
                        default=1)
    parser.add_argument('--genre_unit',
                        type=int,
                        default=5)
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2)
    parser.add_argument('--num_class',
                        type=int,
                        default=3)
    parser.add_argument('--word_embedding',
                        choices=const.WORD2VEC_TYPES,
                        default=const.WORD2VEC_UPDATE)

    parser.add_argument('--objective',
                        choices=['conversion', 'cvr', 'click'],
                        nargs='+',
                        default=['conversion'])

    return parser.parse_args()
