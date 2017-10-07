import tensorflow as tf

from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import ResnetModel

import neural_toolbox.film_layer as film
import neural_toolbox.rnn as rnn
import neural_toolbox.utils as utils
import neural_toolbox.ft_utils as ft_utils

###
###  Still under development!!!
###


class FiLMNetwork(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "clevr", device=device)

        assert True, "Not tested network... still under development"

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            self.batch_size = None

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size], name='answer')

            self._is_training = tf.placeholder(tf.bool, name="is_training")
            self._weight_decay = tf.placeholder_with_default(0. , shape=[],  name="weight_decay")

            word_emb = utils.get_embedding(self._question,
                                           n_words=no_words,
                                           n_dim=config["question"]["word_embedding_dim"],
                                           scope="word_embedding")

            gru_cell = tf.contrib.rnn.GRUCell(
                num_units=config["question"]["rnn_state_size"],
                activation=tf.nn.tanh,
                reuse=reuse)

            self.rnn_state, _ = tf.nn.dynamic_rnn(
                gru_cell,
                word_emb,
                dtype=tf.float32,
                sequence_length=self._seq_length)

            self.rnn_state = rnn.last_relevant(self.rnn_state, self._seq_length)

            #####################
            #   PICTURES
            #####################

            self._picture = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='picture')
            self.picture_out = get_image_features(
                image=self._picture, question=self.rnn_state,
                is_training=self._is_training,
                scope_name=scope.name,
                config=config['image']
            )

            assert len(self.picture_out.get_shape()) == 4, \
                "Incorrect image input and/or attention mechanism (should be none)"

            #####################
            #   STEM
            #####################

            with tf.variable_scope("Stem", reuse=reuse):

                stem_features = self.picture_out
                if config["stem"]["spatial_location"]:
                    stem_features = ft_utils.append_spatial_location(stem_features)

                self.stem_conv = tf.contrib.layers.conv2d(stem_features,
                                                          num_outputs=config["stem"]["conv_out"],
                                                          kernel_size=config["stem"]["conv_kernel"],
                                                          activation_fn=None,
                                                          reuse=reuse)

                self.stem_out = tf.layers.batch_normalization(self.stem_conv)
                self.stem_out = tf.nn.relu(self.stem_out)

            #####################
            #   FiLM Layers
            #####################

            with tf.variable_scope("ResBlocks", reuse=reuse):

                res_output = self.stem_out
                self.resblocks = []

                for i in range(config["resblock"]["no_resblock"]):
                    with tf.variable_scope("ResBlock_{}".format(i), reuse=reuse):
                        resblock = film.FiLMResblock(res_output, self.rnn_state,
                                                     kernel1=config["resblock"]["kernel1"],
                                                     kernel2=config["resblock"]["kernel2"],
                                                     spatial_location=config["resblock"]["spatial_location"],
                                                     is_training=self._is_training,
                                                     reuse=reuse)

                        self.resblocks.append(resblock)
                        res_output = resblock.get()

            #####################
            #   Classifier
            #####################

            with tf.variable_scope("classifier", reuse=reuse):

                classif_features = res_output
                if config["classifier"]["spatial_location"]:
                    classif_features = ft_utils.append_spatial_location(classif_features)

                # 2D-Conv
                self.classif_conv = tf.contrib.layers.conv2d(classif_features,
                                                             num_outputs=config["classifier"]["conv_out"],
                                                             kernel_size=config["classifier"]["conv_kernel"],
                                                             activation_fn=None,
                                                             reuse=reuse)
                self.classif_conv = tf.layers.batch_normalization(self.classif_conv)
                self.classif_conv = tf.nn.relu(self.classif_conv)


                self.max_pool = tf.contrib.layers.max_pool2d(self.classif_conv,
                                                             self.classif_conv.get_shape()[1:3])
                self.max_pool = tf.reshape(self.max_pool, shape=[-1, int(self.classif_conv.get_shape()[-1])])

                self.hidden_state = tf.contrib.layers.fully_connected(self.max_pool,
                                                               num_outputs=config["classifier"]["no_mlp_units"],
                                                               activation_fn=None,
                                                               reuse=reuse)

                self.hidden_state = tf.layers.batch_normalization(self.hidden_state, axis=1)
                self.hidden_state = tf.nn.relu(self.hidden_state)

                self.out = tf.contrib.layers.fully_connected(self.hidden_state,
                                                      num_outputs=no_answers,
                                                      activation_fn=None,
                                                      reuse=reuse)

            #####################
            #   Loss
            #####################

            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy)

            self.loss_decay = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if self.scope_name in v.name]
            self.loss_decay = self.loss + self._weight_decay * tf.add_n(self.loss_decay)

            self.softmax = tf.nn.softmax(self.out, name='answer_prob')
            self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.prediction, self._answer)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')

if __name__ == "__main__":

    FiLMNetwork({

    "name" : "CLEVR with FiLM",

    "image":
    {
      "image_input": "conv",
      "dim": [14, 14, 1024],
      "normalize": False,

      "attention" : {
        "mode": "none"
      }
    },

    "question": {
      "word_embedding_dim": 200,
      "rnn_state_size": 4096
    },

    "stem" : {
      "spatial_location" : True,
      "conv_out": 128,
      "conv_kernel": [3,3]
    },

    "resblock" : {
      "no_resblock" : 4,
      "spatial_location" :True,
      "kernel1" : [1,1],
      "kernel2" : [3,3]
    },

    "classifier" : {
      "spatial_location" : True,
      "conv_out": 512,
      "conv_kernel": [1,1],
      "no_mlp_units": 1024
    }

  }, no_words=200, no_answers=10)
