"""
https://github.com/FrederikSchorr/sign-language

Train a pre-trained I3D convolutional network to classify videos
"""

import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import keras
from keras import backend as K

from datagenerator import VideoClasses, FramesGenerator, generate_generator_multiple
from model_i3d import Inception_Inflated3d, add_i3d_top, model_fusion
import tensorflow as tf
from keras.models import Model, load_model


def layers_freeze(keModel:keras.Model) -> keras.Model:
    
    print("Freeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = False

    return keModel

def layers_unfreeze(keModel:keras.Model) -> keras.Model:
    
    print("Unfreeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = True

    return keModel


def count_params(keModel:keras.Model):

    #trainable_count = int(
        #np.sum([K.count_params(p) for p in set(keModel.trainable_weights)]))
    #non_trainable_count = int(
        #np.sum([K.count_params(p) for p in set(keModel.non_trainable_weights)]))
    trainable_count = keras.utils.layer_utils.count_params(keModel.trainable_weights)
    non_trainable_count = keras.utils.layer_utils.count_params(keModel.non_trainable_weights)


    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    return



def train_I3D_oflow_end2end(diVideoSet):
    """ 
    * Loads pretrained I3D model, 
    * reads optical flow data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    #sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    #sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model_flow_mirror"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 3}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the ChaLearn classes
    #oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = FramesGenerator(sOflowDir + "/train_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2)
    genFramesVal = FramesGenerator(sOflowDir + "/val_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights='flow_imagenet_and_kinetics',
        #weights='model/20200704-1221-tsl100-oflow-i3d-entire-best.h5',
        input_shape=(diVideoSet["nFramesNorm"], 224, 224, 2))
    print("Add top layers with %d output classes ..." % 63)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, 63, dropout_prob=0.5)
        
    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-%03d-oflow-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    
    # Helper: Save results
    csv_logger = tf.keras.callbacks.CSVLogger("log_flow_mirror/" + sLog + "-acc_above.csv", append = True)

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    cpTopLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-last.h5", verbose = 0)
    cpTopBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-best.h5",
        verbose = 1, save_best_only = True)
    cpAllLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-last.h5", verbose = 0)
    cpAllBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr = diTrainTop["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainTop["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpTopLast, cpTopBest])
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    csv_logger = tf.keras.callbacks.CSVLogger("log_flow_mirror/" + sLog + "-acc_entire.csv", append = True)   
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow) 

    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return
    



def train_I3D_rgb_end2end(diVideoSet, method='rgb'):
    """ 
    * Loads pretrained I3D model, 
    * reads optical flow data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    if method == 'rgb':
        sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    else:
        sImageDir        = f"data-temp/%s/%s/image_{method}"%(diVideoSet["sName"], sFolder)
    #sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    #sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    #sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model_rgb_mirror"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 3}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the ChaLearn classes
    #oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = FramesGenerator(sImageDir + "/train_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 3)
    genFramesVal = FramesGenerator(sImageDir + "/val_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 3)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(diVideoSet["nFramesNorm"], 224, 224, 3))
    print("Add top layers with %d output classes ..." % 63)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, 63, dropout_prob=0.5)
        
    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-%03d-rgb-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    
    # Helper: Save results
    csv_logger = tf.keras.callbacks.CSVLogger("log_rgb_mirror/" + sLog + "-acc_above.csv", append = True)

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    cpTopLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-last.h5", verbose = 0)
    cpTopBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-best.h5",
        verbose = 1, save_best_only = True)
    cpAllLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-last.h5", verbose = 0)
    cpAllBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr = diTrainTop["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainTop["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpTopLast, cpTopBest])
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    csv_logger = tf.keras.callbacks.CSVLogger("log_rgb_mirror/" + sLog + "-acc_entire.csv", append = True)
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return


def train_I3D_combined_end2end(diVideoSet, method='rgb'):
    """ 
    * Loads pretrained I3D model, 
    * reads optical flow data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    if method == 'rgb':
        sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    else:
        sImageDir        = f"data-temp/%s/%s/image_{method}"%(diVideoSet["sName"], sFolder)
    #sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    #sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    sModelDir        = "model_combined_mirror"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 3}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the ChaLearn classes
    #oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain_flow = FramesGenerator(sOflowDir + "/train_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2, bShuffle=False)
    genFramesVal_flow = FramesGenerator(sOflowDir + "/val_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2, bShuffle=False)
    genFramesTrain_rgb = FramesGenerator(sImageDir + "/train_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 3, bShuffle=False)
    genFramesVal_rgb = FramesGenerator(sImageDir + "/val_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 3, bShuffle=False)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights='flow_imagenet_and_kinetics',
        #weights='model/20200704-1221-tsl100-oflow-i3d-entire-best.h5',
        input_shape=(diVideoSet["nFramesNorm"], 224, 224, 2))
    print("Add top layers with %d output classes ..." % 63)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, 63, dropout_prob=0.5, late_fusion=True)



    print("Load pretrained I3D rgb model ...")
    keI3Drgb = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        #weights='model/20200704-1221-tsl100-oflow-i3d-entire-best.h5',
        input_shape=(diVideoSet["nFramesNorm"], 224, 224, 3),
        layer_name='RGB')
    print("Add top layers with %d output classes ..." % 63)
    keI3Drgb = layers_freeze(keI3Drgb)
    keI3Drgb = add_i3d_top(keI3Drgb, 63, dropout_prob=0.5, late_fusion=True, layer_name='RGB')

    keI3Dfusion = model_fusion(keI3Drgb, keI3DOflow)

    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03dclass-%03dframe-combined-%s-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"], diVideoSet["nFramesNorm"], method)
    
    # Helper: Save results
    csv_logger = tf.keras.callbacks.CSVLogger("log_combined_mirror/" + sLog + "-acc_above.csv", append = True)

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    cpTopLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-last.h5", verbose = 0)
    cpTopBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-best.h5",
        verbose = 1, save_best_only = True)
    cpAllLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-last.h5", verbose = 0)
    cpAllBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr = diTrainTop["fLearn"])
    keI3Dfusion.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3Dfusion)    

    train_gen = generate_generator_multiple(genFramesTrain_rgb, genFramesTrain_flow)
    val_gen = generate_generator_multiple(genFramesVal_rgb, genFramesVal_flow)
    
    keI3Dfusion.fit_generator(
        generator = train_gen,
        validation_data = val_gen,
        epochs = diTrainTop["nEpochs"],
        workers = 4,                 
        use_multiprocessing = False,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpTopLast, cpTopBest])
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    csv_logger = tf.keras.callbacks.CSVLogger("log_combined_mirror/" + sLog + "-acc_entire.csv", append = True)   
    keI3Dfusion = layers_unfreeze(keI3Dfusion)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3Dfusion.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3Dfusion) 

    keI3Dfusion.fit_generator(
        generator = train_gen,
        validation_data = val_gen,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = False,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return

def mnodel_fine_tune(diVideoSet, method='rgb'):
   # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    if method == 'rgb':
        sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    else:
        sImageDir        = f"data-temp/%s/%s/image_{method}"%(diVideoSet["sName"], sFolder)
    #sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    #sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    sModelDir        = "model_combined_mirror"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 3}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 5}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the ChaLearn classes
    #oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain_flow = FramesGenerator(sOflowDir + "/train_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2, bShuffle=False)
    genFramesVal_flow = FramesGenerator(sOflowDir + "/val_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2, bShuffle=False)
    genFramesTrain_rgb = FramesGenerator(sImageDir + "/train_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 3, bShuffle=False)
    genFramesVal_rgb = FramesGenerator(sImageDir + "/val_videos", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 3, bShuffle=False)


        # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-%03d-combined-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    
    cpAllLast = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-last.h5", verbose = 0)
    cpAllBest = tf.keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    keI3Dfusion = load_model('model_combined_mirror/20200723-1559-tsl100-115-combined-i3d-entire-best.h5')
    train_gen = generate_generator_multiple(genFramesTrain_rgb, genFramesTrain_flow)
    val_gen = generate_generator_multiple(genFramesVal_rgb, genFramesVal_flow)

    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    csv_logger = tf.keras.callbacks.CSVLogger("log_combined_mirror/" + sLog + "-acc_entire.csv", append = True)   
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3Dfusion.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3Dfusion) 

    keI3Dfusion.fit_generator(
        generator = train_gen,
        validation_data = val_gen,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = False,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return


if __name__ == '__main__':

    """diVideoSet = {"sName" : "ledasila",
        "nClasses" : 21,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (288, 352), # height, width
        "nFpsAvg" : 25,
        "nFramesAvg" : 75,
        "fDurationAvg" : 3.0} # seconds
    """

    
    diVideoSet = {"sName" : "tsl",
    "nClasses" : 100,   # number of classes
    "nFramesNorm" : 115,    # number of frames per video
    "nMinDim" : 240,   # smaller dimension of saved video-frames
    "tuShape" : (600, 480), # height, width
    "nFpsAvg" : 10,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 5.0} # seconds 
    
    #dtype='float16'
    #K.set_floatx(dtype)
    #K.set_epsilon(1e-4) 
    #import os
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""

    
    #train_I3D_rgb_end2end(diVideoSet)
    #train_I3D_oflow_end2end(diVideoSet)
    #train_I3D_combined_end2end(diVideoSet)
    mnodel_fine_tune(diVideoSet, method='bgSub')
