#   lint as: python3`
"""Main file to run concept learning with AwA dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concept_model
import helper
from utils.log import setup_logger
from utils.ood_utils import run_ood_over_batch
from utils.test_utils import get_measures
from utils.stat_utils import multivar_separa 
# from test_baselines import run_eval

from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras.utils as utils
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import os
import argparse
import logging
import numpy as np
import sys
import time


def get_data(bs, args, ood=True):
    """
    prepare data loaders for ID and OOD data (train/test)
    :param bs: batch size
    :ood: whether to load OOD data as well (False for baseline concept learning by Yeh et al.)
    """

    TRAIN_DIR = "data/AwA2/train"
    VAL_DIR = "data/AwA2/val"
    TEST_DIR = "data/AwA2/test"
    if args.out_data == 'MSCOCO':
        OOD_DIR = "data/MSCOCO"
    elif args.out_data == 'augAwA':
        OOD_DIR = "data/AwA2-train-fractals"

    TARGET_SIZE = (224, 224)
    BATCH_SIZE = bs
    BATCH_SIZE_OOD = bs

    print('Loading images through generators ...')
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_loader = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=TARGET_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)

    #print(train_generator.class_indices.items())

    datagen = ImageDataGenerator(rescale=1.0 / 255.)
    val_loader = datagen.flow_from_directory(VAL_DIR,
                                            batch_size=BATCH_SIZE,
                                            target_size=TARGET_SIZE,
                                            class_mode='categorical',
                                            shuffle=False)
    test_loader = datagen.flow_from_directory(TEST_DIR,
                                            batch_size=BATCH_SIZE,
                                            target_size=TARGET_SIZE,
                                            class_mode='categorical',
                                            shuffle=False)
    if ood:
        #numUpdates = int(NUM_TRAIN / BATCH_SIZE) # int(f_train.shape[0] / BATCH_SIZE)
        #NUM_OOD = 31706
        #BATCH_SIZE_OOD = int(NUM_OOD / numUpdates)
        OOD_loader = train_datagen.flow_from_directory(OOD_DIR, #datagen
                                                batch_size=BATCH_SIZE_OOD,
                                                target_size=TARGET_SIZE,
                                                class_mode=None, shuffle=True)
    else:
        OOD_loader = None

    return train_loader, val_loader, test_loader, OOD_loader


def get_class_labels(loader, savepath):
    """
    extract groundtruth class labels from data loader
    :param loader: data loader
    :param savepath: path to the numpy file
    """

    if os.path.exists(savepath):
        y = np.load(savepath)
    else:
        num_data = len(loader.filenames)
        y = []
        for (_, y_batch), _ in zip(loader, range(len(loader))):
            y.extend(y_batch)
       
        np.save(savepath, y)
    return y

def get_args():
    parser = argparse.ArgumentParser(description='concept learning (both baseline and OOD)')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='optimizer')
    parser.add_argument('--thres', type=float, default=0.2, help='threshold for concept scores')
    parser.add_argument('--val_step', type=int, default=2, help='how often to test with validation set during training')
    parser.add_argument('--save_step', type=int, default=2, help='how often to save the topic model during training')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--trained', '-trained', action='store_true', help='default False - whether topic model is trained')
    parser.add_argument('--num_concepts', type=int, default=70, help='number of concepts; parameter for concept learning')
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--name', type=str, required=True, help='directory to save trained topic model and concepts')
    # different options for concept learning objective
    parser.add_argument('--feat_l2', '-feat_l2', action='store_true', help='whether to use ||feat - recovered feat||_2 regularizer') 
    parser.add_argument('--feat_cosine', '-feat_cosine', action='store_true', help='whether to use cosine distance regularizer between feat and recovered feat')
    parser.add_argument('--separability', '-separability', action='store_true', help='whether to use separability regularization')
    parser.add_argument('--coeff_feat', type=float, default=0.1, help='coefficient for loss_l2')
    parser.add_argument('--coeff_cosine', type=float, default=1., help='coefficient for loss_cos')
    parser.add_argument('--coeff_score', type=float, default=1., help='coefficient for loss_score')
    parser.add_argument('--coeff_concept', type=float, default=10., help='coefficient for loss_coherency and loss_similarity')
    parser.add_argument('--coeff_separa', type=float, default=10., help='coefficient for loss_separa')
    parser.add_argument('--num_hidden', type=int, default=2, help='number of hidden layers for mapping g')
    #parameters for OOD detection
    parser.add_argument('--out_data', type=str, choices=['MSCOCO', 'augAwA'], default='MSCOCO', help='Auxiliary OOD Dataset during concept learning')
    parser.add_argument('--ood', '-ood', action='store_true', help='whether to outsource OOD data during concept learning')
    parser.add_argument('--score', type=str, choices=['energy'], default=None, help='OOD detector type')
    parser.add_argument('--temperature_odin', default=1000, type=int, help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float, help='perturbation magnitude for odin')
    parser.add_argument('--temperature_energy', default=1, type=int, help='temperature scaling for energy')


    return parser.parse_args()


def main():

    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    
    #if not os.path.exists(args.output_dir):
    #    os.makedirs(args.output_dir)

    if args.separability:
        args.ood = True
    USE_OOD = args.ood
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    THRESHOLD = args.thres
    trained = args.trained
    N_CONCEPT = args.num_concepts
    offset = args.offset
    topic_modelpath = os.path.join(args.logdir, args.name,'topic_epoch{}.weights.h5'.format(offset))
    #topic_modelpath = os.path.join(args.logdir, args.name,'topic_latest.h5')
    topic_savepath = os.path.join(args.logdir, args.name,'topic_vec_inceptionv3.npy')

    logger = setup_logger(args)

    train_loader, val_loader, test_loader, ood_loader =  get_data(BATCH_SIZE, args, ood=USE_OOD)

    #print(train_generator.class_indices.items())
    #assert ('_OOD', 0) in val_generator.class_indices.items()
    #y_train = get_class_labels(train_loader, savepath='data/Animals_with_Attributes2/y_train.npy')
    y_val = get_class_labels(val_loader, savepath='data/AwA2/y_val.npy')
    y_test = get_class_labels(test_loader, savepath='data/AwA2/y_test.npy')

    # preds_cls_idx = y_test.argmax(axis=-1)
    # idx_to_cls = {v: k for k, v in test_generator.class_indices.items()}
    # preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
    # filenames_to_cls = list(zip(test_generator.filenames, preds_cls))

    # Loads model
    feature_model, predict_model = helper.load_model_inception_new(train_loader, val_loader, \
            batch_size=BATCH_SIZE, input_size=(224,224), pretrain=True, \
            modelname='./results/AwA2/inceptionv3_AwA2.weights.h5', split_idx=-5)

if __name__ == '__main__':
    main()
