# Original source code can be accessed at https://nvlabs.github.io/sionna/examples/Autoencoder.html#Trainable-End-to-end-System:-RL-based-Training
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
from tensorflow.keras.layers import Layer, Dense
from sionna.channel import AWGN
from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims, hard_decisions
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation, LLRs2SymbolLogits
from sionna.utils import sim_ber

tf.random.set_seed(1)

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 10.0
ebno_db_max = 10.0

###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6  # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 1  # Coderate for the outer code
n = 1500  # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol  # Number of modulated baseband symbols per codeword
k = int(n*coderate)  # Number of information bits per codeword

###############################################
# Training configuration
###############################################
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 3000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(128, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training_10dB" # Filename to save the autoencoder weights once RL-based training is done

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
        # y, no = inputs
        #
        # # Using log10 scale helps with the performance
        # no_db = log10(no)
        #
        # # Stacking the real and imaginary components of the complex received samples and the noise variance
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

class E2ESystemRLTraining(Model):

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
            # self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    # @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db, perturbation_variance=tf.constant(0.0, tf.float32)):

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

        # Adding perturbation
        # If ``perturbation_variance`` is 0, then the added perturbation is null
        epsilon_r = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon_i = tf.random.normal(tf.shape(x))*tf.sqrt(0.5*perturbation_variance)
        epsilon = tf.complex(epsilon_r, epsilon_i) # [batch size, num_symbols_per_codeword]
        x_p = x + epsilon # [batch size, num_symbols_per_codeword]

        ################
        ## Channel
        ################
        y = self._channel([x_p, no]) # [batch size, num_symbols_per_codeword]
        y = tf.stop_gradient(y)  # Stop gradient here

        ################
        ## Receiver
        ################
        # llr = self._demapper([y, no])
        llr = self._demapper(y)

        # If training, outer decoding is not performed
        if self._training:
            # Average BCE for each baseband symbol and each batch example
            c = tf.reshape(c, [-1, num_symbols_per_codeword, num_bits_per_symbol])
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(c, llr), axis=2)  # Avergare over the bits mapped to a same baseband symbol
            # The RX loss is the usual average BCE
            rx_loss = tf.reduce_mean(bce)
            # From the TX side, the BCE is seen as a feedback from the RX through which backpropagation is not possible
            bce = tf.stop_gradient(bce)  # [batch size, num_symbols_per_codeword]
            x_p = tf.stop_gradient(x_p)
            p = x_p-x  # [batch size, num_symbols_per_codeword] Gradient is backpropagated through `x`
            tx_loss = tf.square(tf.math.real(p)) + tf.square(tf.math.imag(p))  # [batch size, num_symbols_per_codeword]
            tx_loss = -bce*tx_loss/rl_perturbation_var  # [batch size, num_symbols_per_codeword]
            tx_loss = tf.reduce_mean(tx_loss)
            return tx_loss, rx_loss
        else:
            llr = tf.reshape(llr, [-1, n])  # Reshape as expected by the outer decoder

            llr_sigmoid = tf.sigmoid(llr)
            c_hat = tf.cast(llr_sigmoid>0.5, tf.float32)

            return c, c_hat

def rl_based_training(model):
    # Optimizers used to apply gradients
    optimizer_tx = tf.keras.optimizers.Adam() # For training the transmitter
    optimizer_rx = tf.keras.optimizers.Adam() # For training the receiver

    # Function that implements one transmitter training iteration using RL.
    def train_tx():
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            # Keep only the TX loss
            tx_loss, _ = model(training_batch_size, ebno_db,
                               tf.constant(rl_perturbation_var, tf.float32)) # Perturbation are added to enable RL exploration
        ## Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(tx_loss, weights)
        optimizer_tx.apply_gradients(zip(grads, weights))

    # Function that implements one receiver training iteration
    def train_rx():
        # Sampling a batch of SNRs
        ebno_db = tf.random.uniform(shape=[training_batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        # Forward pass
        with tf.GradientTape() as tape:
            # Keep only the RX loss
            _, rx_loss = model(training_batch_size, ebno_db) # No perturbation is added
        ## Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(rx_loss, weights)
        optimizer_rx.apply_gradients(zip(grads, weights))
        # The RX loss is returned to print the progress
        return rx_loss

    # Training loop.
    for i in range(num_training_iterations_rl_alt):
        # 10 steps of receiver training are performed to keep it ahead of the transmitter
        # as it is used for computing the losses when training the transmitter
        for _ in range(10):
            rx_loss = train_rx()
        # One step of transmitter training
        train_tx()
        # Printing periodically the progress
        if i % 100 == 0:
            print('Iteration {}/{}  BCE {:.4f}'.format(i, num_training_iterations_rl_alt, rx_loss.numpy()), end='\r')
    print() # Line break

    # Once alternating training is done, the receiver is fine-tuned.
    print('Receiver fine-tuning... ')
    for i in range(num_training_iterations_rl_finetuning):
        rx_loss = train_rx()
        if i % 100 == 0:
            print('Iteration {}/{}  BCE {:.4f}'.format(i, num_training_iterations_rl_finetuning, rx_loss.numpy()), end='\r')


def save_weights(model, model_weights_path):
    weights = model.get_weights()
    with open(model_weights_path, 'wb') as f:
        pickle.dump(weights, f)


def main(args=None):
    # Instantiate and train the end-to-end system
    model = E2ESystemRLTraining(training=True)
    rl_based_training(model)
    # Save weights
    save_weights(model, model_weights_path_rl_training)


if __name__ == '__main__':
    main()
