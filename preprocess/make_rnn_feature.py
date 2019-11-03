"""
This module is used to make seq 2 seq features which are [encoder_input_features, decoder_features, decoder_values]
@Author: DuFei
@Created Time: 2019/11/02 19:31
"""

import numpy as np


class InputPipe():
    def __init__(self, input_x, input_y, encoder_length, decoder_length):
        """
        :param input_x: all features input, [sequence_num, sequence_length, feature_length]
        :param input_y: all labels, [data_size, sequence_length, 1]
        :param encoder_length:
        :param decoder_length:
        """
        self.input_x = input_x
        self.input_y = input_y
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.sequence_num = self.input_x.shape[0]
        self.encoder_feature_length = self.input_x.shape[2] + 1
        self.decoder_feature_length = self.input_x.shape[2]

        self.encoder_features_all, self.decoder_features_all, self.decoder_values_all = self.sample()

    def sample(self):
        total_length = self.input_x.shape[1]
        single_sample_length = self.encoder_length + self.decoder_length
        offset = total_length - self.encoder_length - self.decoder_length
        sample_indexes = np.random.randint(0, offset, size=(self.sequence_num, offset))

        total_samples = self.sequence_num * offset

        encoder_features_all = np.zeros((total_samples, self.encoder_length, self.encoder_feature_length))
        decoder_features_all = np.zeros((total_samples, self.decoder_length, self.decoder_feature_length))
        decoder_values_all = np.zeros((total_samples, self.decoder_length, 1))

        sample_index = 0
        for seq_index in range(self.sequence_num):

            for seq_sample_index in sample_indexes[seq_index]:
                x = self.input_x[seq_index][seq_sample_index:seq_sample_index + single_sample_length, :]
                y = self.input_y[seq_index][seq_sample_index:seq_sample_index + single_sample_length, :]

                encoder_feature, decoder_feature = np.split(x, [self.encoder_length])
                encoder_value, decoder_value = np.split(y, [self.encoder_length])

                encoder_features_all[sample_index] = np.concatenate((encoder_feature, encoder_value), -1)
                decoder_features_all[sample_index] = decoder_feature
                decoder_values_all[sample_index] = decoder_value

                sample_index += 1

        return encoder_features_all, decoder_features_all, decoder_values_all
