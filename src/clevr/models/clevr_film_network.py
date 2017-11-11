import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
import tensorflow.contrib.rnn as tfc_rnn

from generic.tf_utils.abstract_network import ResnetModel

import neural_toolbox.film_layer as film
import neural_toolbox.ft_utils as ft_utils

import tensorflow.contrib.slim as slim


class FiLM_CLEVRNetwork(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "clevr", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self.batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size], name='answer')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)

            # word_emb = tf.nn.dropout(word_emb, dropout_keep)
            rnn_cell = tfc_rnn.GRUCell(
                num_units=config["question"]["rnn_state_size"],
                activation=tf.nn.tanh,
                reuse=reuse)

            _, self.rnn_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=word_emb,
                dtype=tf.float32,
                sequence_length=self._seq_length)

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='image')
            assert len(self._image.get_shape()) == 4, \
                "Incorrect image input and/or attention mechanism (should be none)"

            #####################
            #   STEM
            #####################

            with tf.variable_scope("stem", reuse=reuse):

                stem_features = self._image
                if config["stem"]["spatial_location"]:
                    stem_features = ft_utils.append_spatial_location(stem_features)

                self.stem_conv = tfc_layers.conv2d(stem_features,
                                                   num_outputs=config["stem"]["conv_out"],
                                                   kernel_size=config["stem"]["conv_kernel"],
                                                   normalizer_fn=tfc_layers.batch_norm,
                                                   normalizer_params={"center": True, "scale": True,
                                                                    "decay": 0.9,
                                                                    "is_training": self._is_training,
                                                                    "reuse": reuse},
                                                   activation_fn=tf.nn.relu,
                                                   reuse=reuse,
                                                   scope="stem_conv")

                # self.stem_conv = tfc_layers.batch_norm(self.stem_conv, center=True, scale=True, decay=0.9, epsilon=1e-5, is_training=self._is_training, reuse=reuse)
                # self.stem_conv = tf.nn.relu(self.stem_conv)




            #####################
            #   FiLM Layers
            #####################

            with tf.variable_scope("resblocks", reuse=reuse):

                res_output = self.stem_conv
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
                self.classif_conv = tfc_layers.conv2d(classif_features,
                                                      num_outputs=config["classifier"]["conv_out"],
                                                      kernel_size=config["classifier"]["conv_kernel"],
                                                      normalizer_fn=tfc_layers.batch_norm,
                                                      normalizer_params={"center":True, "scale":True,
                                                                         "decay":0.9,
                                                                         "is_training": self._is_training,
                                                                         "reuse": reuse},
                                                      activation_fn=tf.nn.relu,
                                                      reuse=reuse,
                                                      scope="classifier_conv")

                # self.classif_conv = tfc_layers.batch_norm(self.classif_conv, center=True, scale=True, decay=0.9, epsilon=1e-5, is_training=self._is_training, reuse=reuse)
                # self.classif_conv = tf.nn.relu(self.classif_conv)


                self.max_pool = tf.reduce_max(self.classif_conv, axis=[1,2], keep_dims=False, name="global_max_pool")

                self.hidden_state = tfc_layers.fully_connected(self.max_pool,
                                                               num_outputs=config["classifier"]["no_mlp_units"],
                                                               normalizer_fn=tfc_layers.batch_norm,
                                                               normalizer_params={"center": True, "scale": True,
                                                                                  "decay": 0.9,
                                                                                  "is_training": self._is_training,
                                                                                  "reuse": reuse},
                                                               activation_fn=tf.nn.relu,
                                                               reuse=reuse,
                                                               scope="classifier_hidden_layer")

                # self.hidden_state = tf.layers.batch_normalization(self.hidden_state, momentum= 0.1, epsilon = 1e-5, training=self._is_training, reuse=reuse)
                # self.hidden_state = tfc_layers.batch_norm(self.hidden_state, center=True, scale=True, decay=0.9, epsilon=1e-5, is_training=self._is_training, reuse=reuse)
                # self.hidden_state = tf.nn.relu(self.hidden_state)

                self.out = tfc_layers.fully_connected(self.hidden_state,
                                                             num_outputs=no_answers,
                                                             activation_fn=None,
                                                             reuse=reuse,
                                                             scope="classifier_softmax_layer")

            #####################
            #   Loss
            #####################

            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy)

            self.softmax = tf.nn.softmax(self.out, name='answer_prob')
            self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.prediction, self._answer)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":
    FiLM_CLEVRNetwork({

    "name" : "CLEVR with FiLM",

    "image":
    {
      "image_input": "conv",
      "dim": [14, 14, 1024],
      "normalize": False,

      "resnet_out": "block3/unit_22/bottleneck_v1",
      "resnet_version" : 101,
      "attention" : {
        "mode": "none"
      }

    },

    "question": {
      "word_embedding_dim": 200,
      "rnn_cell": "lstm",
      "layer_norm": False,
      "rnn_state_size": 4096
    },

    "stem" : {
      "spatial_location" : True,
      "conv_out": 128,
      "conv_kernel": [3,3]
    },

    "resblock" : {
      "no_resblock" : 4,
      "spatial_location" : True,
      "kernel1" : [1, 1],
      "kernel2" : [3,3]
    },

    "classifier" : {
      "spatial_location" : True,
      "conv_out": 512,
      "conv_kernel": [1,1],
      "no_mlp_units": 1024
    },

    "dropout_keep_prob" : 0.6

  }
    , no_words=354, no_answers=11)
