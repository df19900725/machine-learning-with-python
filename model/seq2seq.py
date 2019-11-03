"""
This module implements Seq2Seq model using by Keras customer layer
@Author: DuFei
@Created Time: 2019/11/03 16:14
"""

import tensorflow as tf


class Seq2Seq(tf.keras.layers.Layer):
    def __init__(self, units, use_lstm=False, use_attention=False, use_bidirectional=False, **kwargs):
        self.units = units
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.use_bidirectional = use_bidirectional

        if self.use_lstm:
            self.encoder_layer = tf.keras.layers.LSTM(units=self.units, return_sequences=True, return_state=True)

            if self.use_bidirectional:
                self.encoder_layer = tf.keras.layers.Bidirectional(self.encoder_layer)
                self.decoder_cell = tf.keras.layers.LSTMCell(units=self.units * 2)
            else:
                self.decoder_cell = tf.keras.layers.LSTMCell(units=self.units)

        else:
            self.encoder_layer = tf.keras.layers.SimpleRNN(units=self.units, return_sequences=True, return_state=True)

            if self.use_bidirectional:
                self.encoder_layer = tf.keras.layers.Bidirectional(self.encoder_layer)
                self.decoder_cell = tf.keras.layers.SimpleRNNCell(units=self.units * 2)
            else:
                self.decoder_cell = tf.keras.layers.SimpleRNNCell(units=self.units)

        self.decoder_project_layer = tf.layers.dense(1, name='decoder_output_projection')

        super(Seq2Seq, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Seq2Seq, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [encoder_features, decoder_features] shape:[batch_size, sequence_length, feature_length]
        :param kwargs:
        :return:
        """
        encoder_feature = inputs[0]
        decoder_feature = inputs[1]

        encoder_output, encoder_states = self.encoder_layer(encoder_feature)
        return self.make_decoder(decoder_feature, encoder_states)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def make_decoder(self, decoder_features, encoder_state):

        # [batch, time_steps, feature_length] -> [time_steps, batch, feature_length]
        input_by_times = tf.transpose(decoder_features, [1, 0, 2])

        decoder_steps = input_by_times.shape[0]
        output = tf.TensorArray(dtype=tf.float32, size=decoder_steps)

        pre_state = encoder_state

        for time_step in range(decoder_steps):
            pre_state, decoder_output = self.decoder_cell(input_by_times[time_step], pre_state)
            output.write(time_step, self.decoder_project_layer(decoder_output))

        return output.stack()
