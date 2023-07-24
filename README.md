# Trainable-Radios-E2E-Communication-Systems
This repository collects some of the open-source codes for reproducing the end-to-end (E2E) communication systems proposed in published papers.

## 1. Trainable Constellation
'E2E_Constellation_Conventional.py' and 'E2E_Constellation_RL.py' are the codes to train the uncoded E2E communication system with trainable constellations under additional white Gaussian noise (AWGN) channel based on the conventional training method [1][2] and reinforcement learning (RL) method [2][3], respectively.

'evaluation_E2E.py' and 'evaluation_RL.py' are the codes to evaluate the bit error rate (BER) of the well-trained models by E2E_Constellation_Conventional.py and E2E_Constellation_RL.py, respectively.

'uncoded_QAM_AWGN.py' realizes an uncoded QAM system under the AWGN channel as a baseline.

These codes are modified from the original source codes at [Trainable-End-to-end-System](https://nvlabs.github.io/sionna/examples/Autoencoder.html#).

###  Environment requirement
'E2E_Constellation_Conventional.py', 'E2E_Constellation_RL.py' and 'uncoded_QAM_AWGN.py' require TensorFlow 2.10 or newer and Python 3.6-3.9, detailed installation requirements of the key python library Sionna can be found at [Sionna Installation](https://nvlabs.github.io/sionna/installation.html).

### Reference
[1] T. O’Shea and J. Hoydis, “An Introduction to Deep Learning for the Physical Layer,” _IEEE Trans. Cogn. Commun. Netw._, vol. 3, no. 4, pp. 563-575, Dec. 2017.

[2] S. Cammerer, F. Ait Aoudia, S. Dörner, M. Stark, J. Hoydis and S. ten Brink, “Trainable Communication Systems: Concepts and Prototype,” _IEEE Trans. Commun._, vol. 68, no. 9, pp. 5489-5503, Sept. 2020.

[3] F. Ait Aoudia and J. Hoydis, “Model-Free Training of End-to-End Communication Systems,” in _IEEE J. on Sel. Areas Commun._, vol. 37, no. 11, pp. 2503-2516, Nov. 2019.



## 2. Trainable Encoder and Decoder
'E2E_conventional_Conv1D.py' and 'E2E_cGAN_Conv1D.py' are the codes to train and evaluate the E2E communication systems with trainable encoders and decoders under AWGN channel with BPSK modulations using conventional training method and conditional generative adversarial network (cGAN) [4], respectively.

The codes are modified from the original source codes at [https://github.com/haoyye/End2End_GAN/blob/master/End2EndConvAWGN.py](https://github.com/haoyye/End2End_GAN/blob/master/End2EndConvAWGN.py) and [https://github.com/moeinheidari/End-to-End-Communications-system/blob/main/E2E-AWGN/Train.py](https://github.com/moeinheidari/End-to-End-Communications-system/blob/main/E2E-AWGN/Train.py), respectively.


###  Environment requirement
'E2E_conventional_Conv1D.py' and 'E2E_cGAN_Conv1D.py' require TensorFlow 2.0 or newer and Tensorflow 1.15 or older, respectively.

### Reference

[4]  H. Ye, L. Liang, G. Y. Li and B. Juang, "Deep learning based end-to-end wireless communication systems with conditional GAN as unknown channel", _IEEE Trans. Wireless Commun._, vol. 19, no. 5, pp. 3133-3143, May 2020.
