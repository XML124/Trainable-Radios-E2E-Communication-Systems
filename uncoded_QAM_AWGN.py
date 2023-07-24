# Original source code can be accessed at https://nvlabs.github.io/sionna/examples/Autoencoder.html#
import tensorflow as tf
import numpy as np
import sionna as sn

# For plotting
import matplotlib.pyplot as plt

# For saving complex Python data structures efficiently
import pickle


# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()

# 64-QAM constellation
NUM_BITS_PER_SYMBOL = 6
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=False) # The constellation is set to be trainable

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation, hard_out=True)
# hard_decision = sn.mapping.SymbolLogits2LLRs("app", NUM_BITS_PER_SYMBOL, hard_out=True)

# AWGN channel
awgn_channel = sn.channel.AWGN()

BATCH_SIZE = 128 # How many examples are processed by Sionna in parallel
EBN0_DB_range = np.arange(-10, 20, 1)

for ebno in EBN0_DB_range:
    no = sn.utils.ebnodb2no(ebno_db=ebno,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here

    bits = binary_source([BATCH_SIZE, 1500])  # Blocklength
    x = mapper(bits)
    y = awgn_channel([x, no])
    llr = demapper([y, no])

    print(np.mean(np.not_equal(bits, llr)))
