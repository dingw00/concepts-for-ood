"""this is where you plug in your OOD detector"""

import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as layers



def get_prediction(predict_model, features):
    logits = predict_model(features)
    if len(logits) == 3: # when TopicModel is used as predict_model
        logits = logits[1]
    return logits


def iterate_data_energy(x, feature_model, predict_model, temper, features=None):
    logits = get_prediction(predict_model, feature_model(x) if features is None else features)
    Ec = -temper * tf.reduce_logsumexp(logits / temper, axis=1)
    
    return -Ec #.numpy()

def iterate_data_msp(x, feature_model, predict_model, features=None):
    logits = get_prediction(predict_model, feature_model(x) if features is None else features)
    Ec = tf.reduce_max(tf.nn.softmax(logits, axis=1), axis=1)
    
    return Ec

def iterate_data_odin(x, feature_model, predict_model, temper, features=None):
    logits = get_prediction(predict_model, feature_model(x) if features is None else features)
    Ec = tf.reduce_max(tf.nn.softmax((logits/temper), axis=1), axis=1)
    
    return Ec

def run_ood_over_batch(x, feature_model, predict_model, args, num_classes, features=None):

    if np.char.lower(args.score) == 'energy':
        scores = iterate_data_energy(x, feature_model, predict_model, args.temperature_energy, features)
    if np.char.lower(args.score) == 'msp':
        scores = iterate_data_msp(x, feature_model, predict_model, features)
    
    if np.char.lower(args.score) == 'odin':
        scores = iterate_data_odin(x, feature_model, predict_model, args.temperature_odin, features)

    return scores #.reshape((0,1))
