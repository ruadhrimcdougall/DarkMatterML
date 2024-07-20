# script libraries
import data
# data handling
import pandas as pd
import numpy as np
# machine learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from hist import Hist
import hist


# load in padded dataset
chonky_padded_waveforms = pd.read_parquet('padded_waveforms.parquet')

# us to timesteps
us2timesteps = 100

# convolution sizes
big_steps = np.flip(np.arange(1*us2timesteps, 13*us2timesteps, us2timesteps))
med_steps = np.flip(np.arange(0.1*us2timesteps, 1*us2timesteps, 0.1*us2timesteps))
smol_steps = np.flip(np.arange(0.01*us2timesteps, 0.1*us2timesteps, 0.01*us2timesteps))
conv_sizes = np.concatenate((big_steps, med_steps, smol_steps))
print(conv_sizes)

# finalising the data arrays
padded_array = chonky_padded_waveforms['chonkers'].to_numpy()
x_data = np.stack(padded_array, axis=0)
# divide by max phd
max_phd = x_data.max()
x_data = x_data / max_phd
# set all values less than zero (probs due to noise) to zero. I think this is the correct way to do this?
x_data[x_data < 0] = 0
# waveform intensity range should now be between zero and 1.
print(x_data.min())
print(x_data.max())
print(x_data.shape)
y_data = chonky_padded_waveforms['label'].to_numpy().reshape((-1,1))
print(y_data.shape)
input_length = x_data.shape[-1]
print(input_length)


# get training and testing sets
runID = chonky_padded_waveforms['runID']
eventID = chonky_padded_waveforms['eventID']
W_array = chonky_padded_waveforms['weights_no_gas'].to_numpy()
X_train, X_test, \
y_train, y_test, \
W_train, W_test, \
runID_train, runID_test, \
eventID_train, eventID_test = \
train_test_split(x_data, y_data, W_array, runID, eventID, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_train)
print(y_train)
X_train_reshaped = np.expand_dims(X_train, axis=-1)
print(X_train_reshaped.shape)


# function to make a cnn
def one_layer_cnn_test(filter_size, num_filters=10):
    CNN_model = keras.Sequential([
        Conv1D(filters=num_filters,kernel_size=filter_size,activation='relu', input_shape=(input_length, 1)),
        Dense(3, activation='softmax')
    ])
    CNN_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                      loss='sparse_categorical_crossentropy',#'binary_crossentropy',
                      weighted_metrics=['accuracy'])
    CNN_model.fit(X_train,y_train,epochs=50, validation_split=0.2, shuffle=True, sample_weight=W_train)
    test_loss, test_acc, test_weight_acc = CNN_model.evaluate(X_test,y_test, sample_weight=W_test)
    return test_loss, test_acc, test_weight_acc

CNN_model = keras.Sequential([
    Masking(mask_value=0., input_shape=(input_length, 1)), # basically means the padded zeros are ignored, unsure exactly how this works
    Conv1D(filters=32,kernel_size=3,activation='relu', padding='valid'),
    Dense(3, activation='softmax')
])
CNN_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                    loss='sparse_categorical_crossentropy',#'binary_crossentropy',
                    weighted_metrics=['accuracy'])
# CNN_model.fit(X_train,y_train,epochs=50, validation_split=0.2, shuffle=True, sample_weight=W_train)
# test_loss, test_acc, test_weight_acc = CNN_model.evaluate(X_test,y_test, sample_weight=W_test)

# loss_lst = []
# acc_lst = []
# w_acc_lst = []
# for i in range(len(conv_sizes)):
#     loss, acc, w_acc = one_layer_cnn_test(int(conv_sizes[i]))
#     loss_lst.append(loss_lst)
#     acc_lst.append(acc)
#     w_acc_lst.append(w_acc)

# plt.figure()
# plt.plot(conv_sizes / us2timesteps, w_acc_lst)
# plt.xlabel('filter sizes (us)')
# plt.ylabel('weighted accuracy')
# plt.title('10 convolutions in 1st layer')
# plt.savefig('first_layer_10_convs.png')