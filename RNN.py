# Author: Milton Candela (https://github.com/miltoncandela)
# Date: August 2026

# The following code trains a Recurrent Neural Network (RNN) using sequential data from https://peerj.com/articles/3298/
# which contains runner's markers positions on the XYZ axis, these markers are attached to multiple joints on the body.
# For each run, a single XYZ force is extracted, and thus source (independent) variables would be the markers positions
# while the target (dependent) variables would be the forces, for each dimension (X, Y, Z).

# WARNING: Numpy version == 1.19.5, otherwise data could not be transformed into a generator and further into RNN:
# NotImplementedError: Cannot convert a symbolic Tensor (simple_rnn/strided_slice:0) to a numpy array.
# This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
# Tensorflow version == 2.5.0, otherwise
# "TypeError: Unable to convert function return value to a Python type! The signature was () -> handle

# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).

# 1.23.5
# previously it was numpy == 1.23.5, scipy == 1.12.0
# now it is numpy == 1.19.5, scipy == 1.6.3
# tensorflow 2.5.0
#

# based on validation df (training) and lets use validation df2 as validation (even though it is testing)


import os
import numpy as np
import pandas as pd
import random
from sys import exit
import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

# from tensorflow.keras.preprocessing import timeseries_dataset_from_array

available_devices = tf.config.experimental.list_physical_devices('GPU')
if len(available_devices) > 0:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_virtual_device_configuration(gpu, [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])  # 2047 MB RTX 3060
        # tf.config.experimental.set_memory_growth(gpu, True)

def get_df(file_name, features_col):
    df = pd.read_csv(file_name)
    # remove non-important features and one-hot encoding of target feature
    df = pd.concat([df[features_col], pd.get_dummies(df['Commanded State'], prefix='State', dtype=int)], axis=1)
    return df

# get training and validation df on selected features
features = ['EAR Avg', 'MAR', 'Head Rot X', 'Head Rot Y', 'Blink Rate']
df_train = get_df('validation_df2.csv', features)
df_valid = get_df('new_validation_df2.csv', features)
print(df_train.shape); print(df_valid.shape)

# StandardScaler considering training dataset
scaler = StandardScaler(); scaler.fit(df_train[features])
scaled_features = pd.DataFrame(scaler.transform(df_train[features]), columns=features)
train = pd.concat([scaled_features, df_train.drop(features,axis=1)], axis=1)
scaled_features = pd.DataFrame(scaler.transform(df_valid[features]), columns=features)
valid = pd.concat([scaled_features, df_valid.drop(features,axis=1)], axis=1)

# Imports tensorflow library, which has deep learning function to build and train a Recurrent Neural Network, further
# code also sets up a GPU with 2GB as a virtual device for faster training, in case the user has one physical GPU.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten

def create_model(model_type, loss_func, model_name=None):
    """
    Using tensorflow and keras, this function builds a sequential model with a RNN architecture, based on layers such
    as SimpleRNN, Dropout and a final Dense layer for the output. When it finishes training, the model is saved on the
    "saved_models" folder when the name parameter is different than None, the function also plots the metrics with
    respect to the number of epochs involve during the computation.

    :param string model_name: Name of the file on which the model would be saved.
    :return object: An already trained RNN, trained using the designated train_generator.
    """

    rnn = Sequential()

    layers_dict = {'LSTM': LSTM(NEURONS, input_shape=(SEQ_SIZE, N_FEATURES), activation='tanh',recurrent_activation='sigmoid',
                     recurrent_dropout=0, unroll=False, use_bias=True),
                   'GRU': GRU(NEURONS, input_shape=(SEQ_SIZE, N_FEATURES)),
                   'Simple': SimpleRNN(NEURONS, input_shape=(SEQ_SIZE, N_FEATURES))}

    if model_type in ['Simple', 'GRU', 'LSTM']:
        rnn.add(layers_dict[model_type])
    elif model_type == 'CNN':
        rnn.add(Conv1D(NEURONS, 3, activation='relu', padding='same', input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(MaxPooling1D(2))
        rnn.add(Conv1D(NEURONS*2, 3, activation='relu', padding='same'))
        rnn.add(MaxPooling1D(2)); rnn.add(Flatten()) # || rnn.add(GlobalAveragePooling1D())
        rnn.add(Dense(NEURONS, activation='relu'))
    elif model_type == 'CNNnopool':
        rnn.add(Conv1D(NEURONS, 3, activation='relu', padding='same', input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Conv1D(NEURONS * 2, 3, activation='relu', padding='same'))
        rnn.add(Flatten())
        rnn.add(Dense(NEURONS, activation='relu'))
    elif model_type == 'CNNsmall':
        rnn.add(Conv1D(NEURONS, 3, activation='relu', padding='same', input_shape=(SEQ_SIZE, N_FEATURES)))
        rnn.add(Flatten()) # || rnn.add(GlobalAveragePooling1D())
    rnn.add(Dense(5, activation='softmax'))

    # Metrics:
    # MeanSquaredError
    # RootMeanSquaredError
    # MeanAbsoluteError
    # MeanAbsolutePercentageError
    # MeanSquaredLogarithmicError
    # CosineSimilarity
    # LogCoshError

    # tf.keras.losses.MeanSquaredError()

    loss_dict = {'MSE': tf.keras.losses.MeanSquaredError(),
                 'MAE': tf.keras.losses.MeanAbsoluteError(),
                 'MSLE': tf.keras.losses.MeanSquaredLogarithmicError()}
    #if loss_func == 'MSE':
    loss = tf.keras.losses.CategoricalCrossentropy()

    rnn.compile(loss=loss, metrics=METRICS.keys(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)) # it used to be 0.000005
    print(rnn.summary())
    history = rnn.fit(train_generator, validation_data=valid_generator, shuffle=False,
                      epochs=EPOCH, verbose=2, batch_size=BATCH_SIZE)
    # callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    if model_name is not None:
        # rnn.save('saved_models/{}_E{}_S{}_B{}.h5'.format(model_name, EPOCH, SEQ_SIZE, BATCH_SIZE))
        rnn.save('{}/model.h5'.format(results_folder_name))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv('{}/csv/history.csv'.format(results_folder_name))

    for metric in list(METRICS.keys()) + ['loss']:
        metrics_fig = plt.figure()
        plt.xlabel('Epoch')
        y_label = METRICS[metric] if metric != 'loss' else METRICS[loss_function.lower()]
        plt.ylabel(y_label)
        plt.title('Training vs Validation {}'.format(metric.upper()))
        plt.plot(hist['epoch'], hist[metric], label='Training')
        plt.plot(hist['epoch'], hist['val_' + metric], label='Validation')
        plt.legend()
        metrics_fig.savefig('{}/figures/E-{}.png'.format(results_folder_name, metric.lower()))

    return rnn

def df_to_generator(df_scaled):
    """
    This function takes a scaled DataFrame and separates the source and target variables into separate DataFrames,
    it also creates an object instance that represents both of the variables into a sequence of size SEQ_SIZE.

    :param pd.DataFrame df: Scaled DataFrame with source and target variables.
    :return (pd.DataFrame, pd.DataFrame, tf.keras.preprocessing.timeseries_dataset_from_array): Separated DataFrames
    depending on whether they have source or target variables, and a generator to train the RNN.
    """

    acc = ['State_0', 'State_1', 'State_2', 'State_3', 'State_4']
    df_output = df_scaled[acc]

    df_scaled = df_scaled.drop(acc, axis=1)
    n_features = df_scaled.shape[1]

    df_generator = TimeseriesGenerator(data=np.array(df_scaled), targets=np.array(df_output),
                                       length=SEQ_SIZE, batch_size=n_features)
    # df_generator = timeseries_dataset_from_array(data=np.array(np.array(df)),
    #                                              targets=np.array(df_output), sequence_length=SEQ_SIZE)

    return df_scaled, df_output, df_generator


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from tensorflow.keras.preprocessing import timeseries_dataset_from_array

# Constants for the RNN
BATCH_SIZE = 128
EPOCH = 30
# 16.67 is 1 second
# 33 2
# 50 3
# 67 4
# 250 15
# 500 30
# states last about 15 seconds, so I guess 15 is the max

for model_type in ['CNNsmall']: # ['CNN', 'CNNnopool', 'CNNsmall']
    for NEURONS in [8]:#[2,4,6,8]:
        for SEED in range(1, 11):
            os.environ['PYTHONHASHSEED'] = str(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            # tf.config.experimental.enable_op_determinism()

            for SEQ_SIZE in [10, 20, 40, 80, 120]: #: # [33,50,83,167,250]: #[1,5,10,20,40,80,120]: # [1,17,33,50,83,167,250]:

                train_scaled, train_output, train_generator = df_to_generator(train)
                valid_scaled, valid_output, valid_generator = df_to_generator(valid)

                # model = tf.keras.models.load_model('saved_models/catcross_E1000_S5_B128.h5')
                # pd.DataFrame(model.predict(train_generator), columns=['S0', 'S1', 'S2', 'S3', 'S4']).to_csv('RNN_train.csv')
                # pd.DataFrame(model.predict(valid_generator), columns=['S0', 'S1', 'S2', 'S3', 'S4']).to_csv('RNN_test.csv')

                # The following chunks of code represents two ways a RNN model could be generated, either by CREATING or IMPORTING,
                # please comment or uncomment the lines of code depending on the desired outcome.

                N_FEATURES = train_scaled.shape[1]
                METRICS = {'mae': 'Mean Absolute Error (MAE)', 'mse': 'Mean Squared Error (MSE)',
                           'msle': 'Mean Squared Logarithmic Error (MSLE)'}

                # create a results_folder_name path
                results_folder_name = 'seqGRU/{}_E{}_B{}_NE{}_S{}_SE{}'.format(model_type, EPOCH, BATCH_SIZE, NEURONS, SEQ_SIZE, SEED)
                if not os.path.exists(results_folder_name):
                    os.mkdir(results_folder_name)
                    os.mkdir(results_folder_name + '/figures')
                    os.mkdir(results_folder_name + '/csv')

                loss_function = 'MAE'; sampling_method = 'up'
                create_model(model_type, loss_function, 'catcross')
