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
from tensorflow.keras.layers import Layer, Dense
from sionna.channel import AWGN
from sionna.utils import sim_ber
from E2E_Constellation_Conventional import E2ESystemConventionalTraining

###############################################
# Evaluation configuration
###############################################

ebno_db_min = -10.0
ebno_db_max = 20.0
num_bits_per_symbol = 6

# Range of SNRs over which the systems are evaluated
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     1) # Step

# Utility function to load and set weights of a model
def load_weights(model, model_weights_path):
    model(1, tf.constant(10.0, tf.float32))
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)


model_weights_path_E2E_training = "awgn_autoencoder_weights_conventional_training_demapper_y" # Filename to save the autoencoder weights once conventional training is done
model_E2E = E2ESystemConventionalTraining(training=False)
load_weights(model_E2E, model_weights_path_E2E_training)
bers, _ = sim_ber(model_E2E, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=100)

ebno_db_test = tf.random.uniform(shape=[2000], minval=5.0, maxval=5.0)
c, c_hat = model_E2E(2000, ebno_db_test)
ber = np.mean(np.not_equal(c, c_hat))
print(ber)


print(bers)
# print(blers)


fig = model_E2E.constellation.show()
fig.suptitle('Conventional training')
fig.savefig('E2E.png')
