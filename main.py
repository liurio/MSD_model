import os
import argparse
import scipy.misc
import numpy as np

import tensorflow as tf
from MSDmodel import *

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--image_size',dest='image_size',type=int,default=128,help='size of input image(applicale to both image A and image B')
parser.add_argument('--fcn_filter_dim',dest='fcn_filter_dim',type=int,default=64,help='# of fcn filters in first conv layer')
parser.add_argument('--A_channels', dest='A_channels', type=int, default=3, help='# of channels of image A')
parser.add_argument('--B_channels', dest='B_channels', type=int, default=3, help='# of channels of image B')

"""Arguments related to run mode"""
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

parser.add_argument('--loss_metric', dest='loss_metric', default='L1', help='L1, or L2')
parser.add_argument('--niter', dest='niter', type=int, default=30, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.00005, help='initial learning rate for adam')#0.0002
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=False, help='if flip the images for data argumentation')
parser.add_argument('--dataset_name', dest='dataset_name', default='celeba', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--lambda_A', dest='lambda_A', type=float, default=20.0, help='# weights of A recovery loss')
parser.add_argument('--lambda_B', dest='lambda_B', type=float, default=20.0, help='# weights of B recovery loss')

parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save the model every save_freq sgd iterations')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = DualNet(sess,image_size=args.image_size,
                        batch_size=args.batch_size,
                        fcn_filter_dim='args.fcn_filter_dim',
                        A_channels=args.A_channels,
                        B_channels=args.B_channels,
                        dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir,
                        lambda_A=args.lambda_A,
                        lambda_B=args.lambda_B,
                        sample_dir=args.sample_dir,
                        loss_metric=args.loss_metric,
                        flip=(args.flip=='True')
                        )
        if args.phase=='train':
            model.train(args)
        if args.phase=='test':
            model.test(args)

if __name__ == '__main__':
    tf.app.run()

