# Jorge Botas Miret and Beatriz Campillo
# Based on DeepDTA paper
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
import random as rn
import pandas as pd

import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from itertools import product
import os


import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers


from datahelper import *
from arguments import argparser, logging
from emetrics import get_aupr, get_cindex, get_rm2




os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

# session_conf = tf.ConfigProto(device_count = { 'CPU' : 40 },
                              # intra_op_parallelism_threads=1,
                              # inter_op_parallelism_threads=40,
                              # allow_soft_placement=True)
# For running on GPU
session_conf = tf.ConfigProto(device_count = { 'CPU' : 6 },
                              allow_soft_placement=True)
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)





figdir = "figures/"
modeldir = "models/"
PID = str(os.getpid())




def combined_method(FLAGS, num_filters, filter_length_smi,
                               filter_length_prot,
                               filter_length_dom):

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    Xdoms = Input(shape=(FLAGS.max_dom_len,), dtype='int32')

    if FLAGS.word_representation:
        smi_embed_input = FLAGS.smi_wordset_size + 1
        seq_embed_input = FLAGS.seq_wordset_size + 1

    else:
        smi_embed_input = FLAGS.charsmiset_size + 1
        seq_embed_input = FLAGS.charseqset_size + 1
    doms_embed_input = FLAGS.domset_size + 1


    # tf.debugging.set_log_device_placement(True)

    # SMILES CNN
    encode_smiles = Embedding(input_dim=smi_embed_input,
                              output_dim=128,
                              input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=num_filters,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=num_filters*2,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=num_filters*3,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    # PROTEIN CNN
    encode_protein = Embedding(input_dim=seq_embed_input,
                               output_dim=128,
                               input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=num_filters,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*2,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*3,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    # PROSITE DOMAINS CNN
    encode_domains = Embedding(input_dim=doms_embed_input,
                               output_dim=128,
                               input_length=FLAGS.max_dom_len)(Xdoms)
    encode_domains = Conv1D(filters=num_filters,
                            kernel_size=filter_length_dom,
                            activation='relu', padding='valid',
                            strides=1)(encode_domains)
    encode_domains = Conv1D(filters=num_filters*2,
                            kernel_size=filter_length_dom,
                            activation='relu', padding='valid',
                            strides=1)(encode_domains)
    encode_domains = Conv1D(filters=num_filters*3,
                            kernel_size=filter_length_dom,
                            activation='relu', padding='valid',
                            strides=1)(encode_domains)
    encode_domains = GlobalMaxPooling1D()(encode_domains)


    # Combine both representations
    encode_interaction = keras.layers.concatenate([encode_smiles,
                                                   encode_protein,
                                                   encode_domains], axis=-1)

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC1 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC1)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # Last FC layer predicts Kd
    predictions = Dense(1, kernel_initializer='normal')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput, Xdoms], outputs=[predictions])

    # Adam optimizer, MSE as loss function
    interactionModel.compile(optimizer='adam',
                             loss='mean_squared_error',
                             metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel,
               to_file='figures/build_combined_categorical.png')

    return interactionModel



def deep_combined_method(FLAGS, num_filters, filter_length_smi,
                               filter_length_prot):

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    if FLAGS.word_representation:
        smi_embed_input = FLAGS.smi_wordset_size + 1
        seq_embed_input = FLAGS.seq_wordset_size + 1

    else:
        smi_embed_input = FLAGS.charsmiset_size + 1
        seq_embed_input = FLAGS.charseqset_size + 1


    # tf.debugging.set_log_device_placement(True)

    # SMILES CNN
    encode_smiles = Embedding(input_dim=smi_embed_input,
                              output_dim=128,
                              input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=num_filters,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=num_filters*2,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=num_filters*3,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    # PROTEIN CNN
    encode_protein = Embedding(input_dim=seq_embed_input,
                               output_dim=128,
                               input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=num_filters,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*2,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*3,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)



    # Combine both representations
    encode_interaction = keras.layers.concatenate([encode_smiles,
                                                   encode_protein], axis=-1)

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC1 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC1)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # Last FC layer predicts Kd
    predictions = Dense(1, kernel_initializer='normal')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    # Adam optimizer, MSE as loss function
    interactionModel.compile(optimizer='adam',
                             loss='mean_squared_error',
                             metrics=[cindex_score])

    print(interactionModel.summary())
    try:
        plot_model(interactionModel,
                   to_file='figures/build_combined_categorical.png')
    except:
        pass

    return interactionModel



def wide_combined_method(FLAGS, num_filters, filter_length_smi,
                               filter_length_prot,
                               filter_length_dom):

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')
    Xdoms = Input(shape=(FLAGS.max_dom_len,), dtype='int32')

    if FLAGS.word_representation:
        smi_embed_input = FLAGS.smi_wordset_size + 1
        seq_embed_input = FLAGS.seq_wordset_size + 1

    else:
        smi_embed_input = FLAGS.charsmiset_size + 1
        seq_embed_input = FLAGS.charseqset_size + 1
    doms_embed_input = FLAGS.domset_size + 1


    # tf.debugging.set_log_device_placement(True)

    # SMILES CNN
    encode_smiles = Embedding(input_dim=smi_embed_input,
                              output_dim=128,
                              input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=num_filters,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=num_filters*2,
                           kernel_size=filter_length_smi,
                           activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    # PROTEIN CNN
    encode_protein = Embedding(input_dim=seq_embed_input,
                               output_dim=128,
                               input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=num_filters,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=num_filters*2,
                            kernel_size=filter_length_prot,
                            activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    # PROSITE DOMAINS CNN
    encode_domains = Embedding(input_dim=doms_embed_input,
                               output_dim=128,
                               input_length=FLAGS.max_dom_len)(Xdoms)
    encode_domains = Conv1D(filters=num_filters,
                            kernel_size=filter_length_dom,
                            activation='relu', padding='valid',
                            strides=1)(encode_domains)
    encode_domains = Conv1D(filters=num_filters*2,
                            kernel_size=filter_length_dom,
                            activation='relu', padding='valid',
                            strides=1)(encode_domains)
    encode_domains = GlobalMaxPooling1D()(encode_domains)


    # Combine both representations
    encode_interaction = keras.layers.concatenate([encode_smiles,
                                                   encode_protein,
                                                   encode_domains], axis=-1)

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC1 = Dropout(0.3)(FC1)
    FC2 = Dense(1024, activation='relu')(FC1)
    FC2 = Dropout(0.3)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # Last FC layer predicts Kd
    predictions = Dense(1, kernel_initializer='normal')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput, Xdoms], outputs=[predictions])

    # Adam optimizer, MSE as loss function
    interactionModel.compile(optimizer='adam',
                             loss='mean_squared_error',
                             metrics=[cindex_score])

    print(interactionModel.summary())
    try:
        plot_model(interactionModel,
                   to_file='figures/build_combined_categorical.png')
    except:
        pass

    return interactionModel



def prepare_interaction_pairs(FLAGS, XD, XT, Xdoms, Y, rows, cols):
    """"
    XD: matrix containing drug SMILES
    XT: matrix containing target protein sequences
    Y: matrix containing drug-target affinity
    rows: indices for drugs
    cols: indices for targets
    return: SMILES, sequences, affinity
    """
    drugs = []
    targets = []
    doms = []
    affinity=[]

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)
        if FLAGS.extract_domains:
            dom = Xdoms[cols[pair_ind]]
            doms.append(dom)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)
    doms_data = np.stack(doms)

    return drug_data, target_data, doms_data, affinity



def run_kiba_model(XD, XT, Xdoms,  Y, label_row_inds, label_col_inds, perfmeasure,
              deepmethod, FLAGS, train_set, test_set):
    """
    Train both CNN blocks (SMILES and target sequences) and DeepDTA without cross-validation
    model saved to ${pid}model.json
    weights saved to ${pid}model.h5
    predictions saved to predicted_affinities.txt
    correct affinities for test set saved to correct_affinities.txt
    """
    num_filters = FLAGS.num_filters                              # 32
    lenfilter_smi = FLAGS.smi_filter_length                     # 4 or 8
    lenfilter_prot = FLAGS.seq_filter_length                    #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch                                      #100
    batchsz = FLAGS.batch_size


    # Retrieve training set SMILES, sequences and affinities
    trrows = label_row_inds[train_set]
    trcols = label_col_inds[train_set]

    train_drugs, train_prots, train_doms, train_Y = prepare_interaction_pairs(
                                                                   FLAGS,
                                                                   XD, XT,
                                                                   Xdoms, Y,
                                                                   trrows, trcols
                                                                   )

    # Retrieve test set SMILES, sequences and affinities
    testrows = label_row_inds[test_set]
    testcols = label_col_inds[test_set]

    test_drugs, test_prots, test_doms, test_Y = prepare_interaction_pairs(
                                                               FLAGS,
                                                               XD, XT, 
                                                               Xdoms,Y,
                                                               testrows, testcols
                                                               )

    if FLAGS.extract_domains:
        lenfilter_dom = FLAGS.dom_filter_length

        # Build the combined model
        model = wide_combined_method(FLAGS, num_filters, lenfilter_smi,
                                lenfilter_prot, lenfilter_dom)
        # Keras callback function
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

        # TRAINING the combined model
        logging("\n---TRAINING-----\n", FLAGS)
        gridres = model.fit(( [np.array(train_drugs), np.array(train_prots),
                               np.array(train_doms)] ),
                                np.array(train_Y), batch_size=batchsz,
                                epochs=epoch, shuffle=False,
                                validation_split=0.2, callbacks=[es])
    else:

        # Build the combined model
        model = deepmethod(FLAGS, num_filters, lenfilter_smi,
                                lenfilter_prot)
        # Keras callback function
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

        # TRAINING the combined model
        logging("\n---TRAINING-----\n", FLAGS)
        gridres = model.fit(( [np.array(train_drugs), np.array(train_prots)] ),
                                np.array(train_Y), batch_size=batchsz,
                                epochs=epoch, shuffle=False,
                                validation_split=0.2, callbacks=[es])


    try:
        modelname = modeldir+PID+"model"
        model.save(modelname)
        print("Model saved at: " + modelname)
    except:
        print("Unable to save trained model")

    # PREDICTION
    logging("\n---TESTING-----\n", FLAGS)
    if FLAGS.extract_domains:
        predicted_labels = model.predict( [np.array(test_drugs),
            np.array(test_prots), np.array(test_doms)] )
    else:
        predicted_labels = model.predict( [np.array(test_drugs),
            np.array(test_prots)])

    # Save predictions and correct affinities

    try:
        predicted = pd.DataFrame(predicted_labels)
        predicted.to_csv(modeldir+PID+"-predicted_affinities.txt", "w")
        correct = pd.DataFrame(test_Y)
        correct.to_csv(modeldir+PID+"-correct_affinities.txt", "w")
    except:
        print("\nUnable to save predictions\n\n")
        print(predicted_labels)
        print('\n\n')
        print(test_Y)

    # Evaluation metrics for the model
    if FLAGS.extract_domains:
        loss, rperf2 = model.evaluate(([np.array(test_drugs),
                                        np.array(test_prots),
                                        np.array(test_doms)]),
                                        np.array(test_Y), verbose=0)
    else:
        loss, rperf2 = model.evaluate(([np.array(test_drugs),np.array(test_prots)]),
                                        np.array(test_Y), verbose=0)

    rperf = perfmeasure(test_Y, predicted_labels)
    rperf = rperf[0]


    logging("CI-i = %f, CI-ii = %f, MSE = %f" %
            (rperf, rperf2, loss), FLAGS)

    try:
        plotLoss(gridres, num_filters, lenfilter_smi, lenfilter_prot)
    except:
        pass

    return rperf, loss



def run_model(FLAGS, perfmeasure, deepmethod, XDtrain, XTtrain, Xdomtrain, Ytrain,
                 XDtest, XTtest, Xdomtest, Ytest):
    """
    Train both CNN blocks (SMILES and target sequences) and DeepDTA without cross-validation
    model saved to ${pid}model.json
    weights saved to ${pid}model.h5
    predictions saved to predicted_affinities.txt
    correct affinities for test set saved to correct_affinities.txt
    """

    num_filters = FLAGS.num_filters                              # 32
    lenfilter_smi = FLAGS.smi_filter_length                     # 4 or 8
    lenfilter_prot = FLAGS.seq_filter_length                    #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    lenfilter_dom = FLAGS.dom_filter_length
    epochs = FLAGS.num_epoch                                      #100
    batchsz = FLAGS.batch_size


    # Build the combined model
    model = deepmethod(FLAGS, num_filters, lenfilter_smi,
                            lenfilter_prot, lenfilter_dom)
    # Keras callback function
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

    # TRAINING the combined model
    logging("\n---TRAINING-----\n", FLAGS)
    gridres = model.fit(( [np.array(XDtrain), np.array(XTtrain),
                           np.array(Xdomtrain)] ),
                           np.array(Ytrain), batch_size=batchsz,
                           epochs=epochs, shuffle=False,
                           validation_split=0.2, callbacks=[es])

    # Save trained model for later training or testing
    try:
        modelname = modeldir+PID+"model"
        model.save(modelname)
        print("Model save at: " + modelname)
    except:
        print("\nUnable to save trained model\n")

    # PREDICTION
    logging("\n---TESTING-----\n", FLAGS)
    print("\n---TESTING-----\n")
    predicted_labels = model.predict( [np.array(XDtest), np.array(XTtest),
                                       np.array(Xdomtest)] )

    # Save predictions and correct affinities
    try:
        predicted = pd.DataFrame(predicted_labels)
        predicted.to_csv(modeldir+PID+"-predicted_affinities.txt", "w")
        Ytest.to_csv(modeldir+PID+"-correct_affinities.txt", "w")
    except:
        print("\nUnable to save predictions\n\n")
        print(predicted_labels)
        print('\n\n')
        print(Ytest)


    # Evaluation metrics for the model
    loss, rperf2 = model.evaluate(([np.array(XDtest),np.array(XTtest),
                                    np.array(Xdomtest)]),
                                    np.array(Ytest), verbose=0)
    rperf = perfmeasure(Ytest, predicted_labels)
    rperf = rperf[0]


    logging("CI-i = %f, CI-ii = %f, MSE = %f" %
            (rperf, rperf2, loss), FLAGS)

    plotLoss(gridres, num_filters, lenfilter_smi, lenfilter_prot)




    return rperf, loss



def cindex_score(y_true, y_pred):
    """
    Compute concordance index
    """

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)



def plotLoss(history, batchind, epochind, param3ind):

    dataset = str(FLAGS.dataset_path).split("/")[-1]
    figname = dataset
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
	#plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+figname +"_loss.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()


    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/"+figname + "_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()

    # Save history to plot externally
    try:
        with open(modeldir+PID+"loss.txt", "w") as loss:
            loss.writelines(history.history['loss'])

        with open(modeldir+PID+"val_loss.txt", "w") as val_loss:
            val_loss.writelines(history.history['loss'])
    except:
        print("Error saving loss")



def kiba_experiment(FLAGS, perfmeasure, deepmethod):
    """
    Train deepDTA on dataset in FLAGS
    No cross-validation

    perfmeasure: function
        input: lists of correct and predicted labels
        output: performance
            Higher values should show better performance
            e.g. Concordance Index (cindex_score)
            e.g. Inverse of error
    deepmethod: combined CNN for drug and target encoding + deepDTA\n\
                function: combined_method()
    """
    ### Input
    # XD: [drugs, features] sized array (features may also be similarities with
    #   other drugs)
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries


    dataset = DataSet( path = FLAGS.dataset_path,
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      word_representation = FLAGS.word_representation,
                      seq_wordlen = FLAGS.seq_wordlen,
                      smi_wordlen = FLAGS.smi_wordlen,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Xdoms, Y = dataset.parse_kiba(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print("Drug count: " + str(drugcount))
    targetcount = XT.shape[0]
    print("Target count: " + str(targetcount))

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    # Find point address for affinity [x, y]
    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)


    train_set, test_set = dataset.read_sets(FLAGS)

    run_kiba_model(XD, XT, Xdoms, Y, label_row_inds, label_col_inds, perfmeasure,
              deepmethod, FLAGS, train_set, test_set)



def experiment(FLAGS, permeasure, deepmethod):
    """
    Train deepDTA on dataset in FLAGS
    No cross-validation

    perfmeasure: function
        input: lists of correct and predicted labels
        output: performance
            Higher values should show better performance
            e.g. Concordance Index (cindex_score)
            e.g. Inverse of error
    deepmethod: combined CNN for drug and target encoding + deepDTA\n\
                function: combined_method()
    """

    dataset = DataSet( path = FLAGS.dataset_path,
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      word_representation = FLAGS.word_representation,
                      seq_wordlen = FLAGS.seq_wordlen,
                      smi_wordlen = FLAGS.smi_wordlen,
                      need_shuffle = False )

    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size


    XDtrain, XTtrain, Xdomtrain, Ytrain, XDtest, XTtest, Xdomtest, Ytest = dataset.parse_data(FLAGS)

    run_model(FLAGS, perfmeasure, deepmethod, XDtrain, XTtrain, Xdomtrain, Ytrain,
              XDtest, XTtest, Xdomtest, Ytest)



if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + PID + "/"

    dirs = [modeldir, figdir, FLAGS.log_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    print("Log directory: " + str(FLAGS.log_dir))

    print("pid: " + PID)

    logging(str(FLAGS), FLAGS)

    perfmeasure = get_cindex
    deepmethod = wide_combined_method

    if FLAGS.dataset_path == "data/kiba/":
        kiba_experiment(FLAGS, perfmeasure, deep_combined_method)
    else:
        experiment(FLAGS, perfmeasure, deepmethod)
