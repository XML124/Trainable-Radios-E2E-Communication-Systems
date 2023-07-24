# Original source code can be accessed at https://github.com/moeinheidari/End-to-End-Communications-system/blob/main/E2E-AWGN/Train.py
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, Flatten, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)
'''
 --- COMMUNICATION PARAMETERS ---
'''


def BLER_func(y_true, y_pred):
    WER_R_BLER = 1 - tf.reduce_mean(tf.cast(tf.reduce_all(tf.squeeze(tf.abs(y_pred - y_true) < 0.5), axis=-1), tf.float32))
    return WER_R_BLER

def channel_layer(x, sigma):
    w = tf.random.normal(tf.shape(x), mean=0.0, stddev=sigma)
    return x + w


class E2EConventionalLearning(tf.keras.Model):
    def __init__(self, n, noise_sigma, **kwargs):
        super().__init__()

        self.noise_sigma = noise_sigma
        self.n = n


        self.Conv1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')  #default channel last, input shape=(batch, block_length, 1)
        self.Conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.Conv3 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.Conv4 = Conv1D(filters=2, kernel_size=3, padding='same') # output shape=(batch, block_length, 2)


        self.Conv1_d = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')

        self.Conv2_d_ori = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')
        self.Conv2_d = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')
        self.Conv2_d_1 = Conv1D(filters=64, kernel_size=3, padding='same')

        self.Conv3_d_ori = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')
        self.Conv3_d = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')
        self.Conv3_d_1 = Conv1D(filters=64, kernel_size=3, padding='same')

        self.Conv4_d = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.Conv5_d = Conv1D(filters=1, kernel_size=3, padding='same', activation='sigmoid')



    def call(self, inputs):

        e = self.Conv1(inputs)
        e = self.Conv2(e)
        e = self.Conv3(e)
        e = self.Conv4(e)
        layer_4_normalized = tf.scalar_mul(tf.sqrt(tf.cast(self.n, tf.float32)), tf.nn.l2_normalize(e, axis=1))  # normalize the encoding.

        # AWGN channel
        y_h = Lambda(channel_layer, arguments={'sigma': self.noise_sigma}, name='channel_layer')(layer_4_normalized)

        d = self.Conv1_d(y_h)

        d_conv2_ori = self.Conv2_d_ori(d)

        d_conv2 = self.Conv2_d(d_conv2_ori)
        d_conv2 = self.Conv2_d_1(d_conv2)
        d = d_conv2_ori + d_conv2
        d = tf.nn.relu(d)

        d_conv3_ori = self.Conv3_d_ori(d)

        d_conv3 = self.Conv3_d(d_conv3_ori)
        d_conv3 = self.Conv3_d_1(d_conv3)
        d = d_conv3_ori + d_conv3
        d = tf.nn.relu(d)

        d = self.Conv4_d(d)

        Decoding_logit = self.Conv5_d(d)

        return Decoding_logit


# Bits per message
k = 4
# Channel Use
n = 8
# Effective Throughput
#  bits per message / channel use
R = k / n
# Eb/N0 used for training
train_Eb_dB = 3
# Noise Standard Deviation
noise_sigma_train = np.sqrt(1 / (2 * R * 10 ** (train_Eb_dB / 10)))


# Number of messages used for training, each size = k*L
batch_size = 320
batch_size_test = 320
nb_train_word = batch_size*1000
nb_test_word = batch_size*500


'''
 --- GENERATING INPUT DATA ---
'''
# Generate training binary Data
data = np.random.binomial(1, 0.5, [nb_train_word, k, 1]).astype(float)
label = data


'''
 --- NEURAL NETWORKS PARAMETERS ---
'''
early_stopping_patience = 4
epochs = 100
optimizer = Adam(lr=0.0001)


sys_model = E2EConventionalLearning(n, noise_sigma_train)
_ = sys_model(tf.ones([1, k, 1]))
sys_model.summary()
sys_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[BLER_func])

# Save the best results based on Training Set
checkpoint_path = './' + 'model_E2E_Conv1D_' + str(k) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
modelcheckpoint = ModelCheckpoint(filepath=checkpoint_path,
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)
# Early stopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)
# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)

print('starting train the NN...')
start = time.time()

# TRAINING
mod_history = sys_model.fit(data, label,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.3, callbacks=[early_stopping, modelcheckpoint, reduce_lr])

end = time.time()
print('The NN has trained ' + str(end - start) + ' s')



SNR_test = np.arange(-10, 20, 1)
std_list = [np.sqrt(1 / (2 * R * 10 ** (snr_i / 10))) for snr_i in SNR_test]
BLER_result = []
for std in std_list:
    # Generate training binary Data
    test_data = np.random.randint(low=0, high=2, size=(nb_test_word, k, 1)).astype(float)
    test_label = test_data

    sys_model_test = E2EConventionalLearning(n, std)
    _ = sys_model_test(tf.ones([1, k, 1]))
    sys_model_test.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[BLER_func])
    sys_model_test.load_weights(checkpoint_path)

    [_, bler] = sys_model_test.evaluate(test_data, test_label, verbose=0)
    BLER_result.append(bler)

print(BLER_result)

plt.semilogy(SNR_test, BLER_result, label='E2EConventional-Conv1D')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('SNR(dB)')
plt.ylabel('Block error rate')
plt.savefig('Result_E2EConventional-Conv1D.png')
plt.show()
