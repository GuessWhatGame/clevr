import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.attention_factory import get_attention

import neural_toolbox.rnn as rnn

from neural_toolbox.film_stack import FiLM_Stack


class FiLMCLEVRNetwork(ResnetModel):

    def __init__(self, config, num_words, num_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "clevr", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):
            batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config["dropout_keep_prob"])
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [batch_size], name='answer')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=num_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)

            if config["question"]['glove']:
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)

            word_emb = tf.nn.dropout(word_emb, dropout_keep)

            _, last_rnn_state = rnn.rnn_factory(
                inputs=word_emb,
                seq_length=self._seq_length,
                cell=config["question"]["cell"],
                num_hidden=config["question"]["rnn_state_size"],
                bidirectional=config["question"]["bidirectional"],
                max_pool=config["question"]["max_pool"],
                layer_norm=config["question"]["layer_norm"],
                reuse=reuse)

            last_rnn_state = tf.nn.dropout(last_rnn_state, dropout_keep)

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

            visual_features = get_image_features(image=self._image,
                                                 is_training=self._is_training,
                                                 config=config['image'])

            with tf.variable_scope("image_film_stack", reuse=reuse):
                film_stack = FiLM_Stack(image=visual_features,
                                        film_input=last_rnn_state,
                                        is_training=self._is_training,
                                        config=config["film_block"],
                                        reuse=reuse)

                visual_features = film_stack.get()

            # Pool Image Features
            with tf.variable_scope("image_pooling"):
                multimodal_features = get_attention(visual_features, last_rnn_state,
                                                    is_training=self._is_training,
                                                    config=config["pooling"],
                                                    dropout_keep=dropout_keep,
                                                    reuse=reuse)

            with tf.variable_scope("classifier"):
                self.hidden_state = tfc_layers.fully_connected(multimodal_features,
                                                               num_outputs=config["classifier"]["no_mlp_units"],
                                                               normalizer_fn=tfc_layers.batch_norm,
                                                               normalizer_params={"center": True, "scale": True,
                                                                                  "decay": 0.9,
                                                                                  "is_training": self._is_training,
                                                                                  "reuse": reuse},
                                                               activation_fn=tf.nn.relu,
                                                               reuse=reuse,
                                                               scope="classifier_hidden_layer")

                self.out = tfc_layers.fully_connected(self.hidden_state,
                                                      num_outputs=num_answers,
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

    import json
    with open("../../../config/clevr/config.film.json", 'r') as f_config:
        conf = json.load(f_config)

    FiLMCLEVRNetwork(conf["model"], num_words=354, num_answers=56)
