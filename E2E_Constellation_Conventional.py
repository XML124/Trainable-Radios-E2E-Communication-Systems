# Original source code can be accessed at https://nvlabs.github.io/sionna/examples/Autoencoder.html#Trainable-End-to-end-System:-Conventional-Training
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Index of the GPU to use
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, GaussianNoise
from sionna.channel import AWGN
from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims, hard_decisions
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.utils import sim_ber

# Fix the seed for reproducible trainings
tf.random.set_seed(1)

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 10.0
ebno_db_max = 10.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 1 # Coderate for the outer code
n = 1500 # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword

###############################################
# Training configuration
###############################################
num_training_iterations_conventional = 10000 # Number of training iterations for conventional training
training_batch_size = tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training = "awgn_autoencoder_weights_conventional_training_demapper_y" # Filename to save the autoencoder weights once conventional training is done
###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results" # Location to save the results


class NeuralDemapper(Layer):

    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        self._dense_3 = Dense(num_bits_per_symbol, None) # The feature correspond to the LLRs for every bits carried by a symbol

    def call(self, inputs):
        # y,no = inputs
        #
        # # Using log10 scale helps with the performance
        # no_db = log10(no)
        #
        # # Stacking the real and imaginary components of the complex received samples
        # # and the noise variance
        # no_db = tf.tile(no_db, [1, num_symbols_per_codeword]) # [batch size, num_symbols_per_codeword]
        # z = tf.stack([tf.math.real(y),
        #               tf.math.imag(y),
        #               no_db], axis=2) # [batch size, num_symbols_per_codeword, 3]

        y = inputs
        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y)], axis=2)

        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        llr = self._dense_3(llr) # [batch size, num_symbols_per_codeword, num_bits_per_symbol]

        return llr

class E2ESystemConventionalTraining(Model):

    def __init__(self, training):
        super().__init__()

        self._training = training

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        # if not self._training:
        #     # num_bits_per_symbol is required for the interleaver
        #     self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        # Trainable constellation
        constellation = Constellation("qam", num_bits_per_symbol, trainable=True)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)

        ################
        ## Channel
        ################
        self._channel = AWGN()

        ################
        ## Receiver
        ################
        # We use the previously defined neural network for demapping
        self._demapper = NeuralDemapper()
        # To reduce the computational complexity of training, the outer code is not used when training,
        # as it is not required
        # if not self._training:
        #     self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

        #################
        # Loss function
        #################
        if self._training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        if self._training:
            c = self._binary_source([batch_size, n])
        else:
            # b = self._binary_source([batch_size, k])
            # c = self._encoder(b)
            c = self._binary_source([batch_size, n])
        # Modulation
        x = self._mapper(c) # x [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self._channel([x, no]) # [batch size, num_symbols_per_codeword]

        ################
        ## Receiver
        ################
        # llr = self._demapper([y, no])
        llr = self._demapper(y)
        llr = tf.reshape(llr, [batch_size, n])
        # If training, outer decoding is not performed and the BCE is returned
        if self._training:
            loss = self._bce(c, llr)
            return loss
        else:
            # Outer decoding
            # b_hat = self._decoder(llr)
            llr_sigmoid = tf.sigmoid(llr)
            b_hat = tf.cast(llr_sigmoid>0.5, tf.float32)

            return c, b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation


def conventional_training(model):
    # Optimizer used to apply gradients
    optimizer = tf.keras.optimizers.Adam()

    for i in range(num_training_iterations_conventional):
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            loss = model(training_batch_size, ebno_db) # The model is assumed to return the BMD rate
        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        # Printing periodically the progress
        if i % 100 == 0:
            print('Iteration {}/{}  BCE: {:.4f}'.format(i, num_training_iterations_conventional, loss.numpy()), end='\r')


def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)


def main(args=None):

    # Fix the seed for reproducible trainings
    tf.random.set_seed(1)
    # Instantiate and train the end-to-end system
    model = E2ESystemConventionalTraining(training=True)
    conventional_training(model)
    # Save weights
    save_weights(model, model_weights_path_conventional_training)


if __name__ == '__main__':
    main()
