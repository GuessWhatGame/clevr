import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from neural_toolbox import rnn

from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features, get_cbn
from generic.tf_factory.attention_factory import get_attention
from generic.tf_factory.fusion_factory import get_fusion_mechanism


class CLEVRNetwork(ResnetModel):

    def __init__(self, config, num_words, num_answers, device='', reuse=False):
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
            self._answer = tf.placeholder(tf.int64, [batch_size, num_answers], name='answer')

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

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')

            cbn = None
            if "cbn" in config:
                cbn = get_cbn(config["cbn"], last_rnn_state, dropout_keep, self._is_training)

            self.image_out = get_image_features(image=self._image,
                                                is_training=self._is_training,
                                                config=config['image'],
                                                cbn=cbn)

            if len(self.image_out.get_shape()) > 2:
                with tf.variable_scope("image_pooling"):
                    self.image_out = get_attention(self.image_out, last_rnn_state,
                                                   is_training=self._is_training,
                                                   config=config["pooling"],
                                                   dropout_keep=dropout_keep,
                                                   reuse=reuse)

            #####################
            #   FUSION
            #####################

            self.visdiag_embedding = get_fusion_mechanism(input1=self.image_out,
                                                          input2=last_rnn_state,
                                                          config=config.get["fusion"],
                                                          dropout_keep=dropout_keep)

            #####################
            #   CLASSIFIER
            #####################

            with tf.variable_scope('mlp'):
                num_hiddens = config['classifier']['no_mlp_units']

                self.out = tfc_layers.fully_connected(self.visdiag_embedding, num_hiddens, activation_fn=tf.nn.relu)
                self.out = tfc_layers.fully_connected(self.out, num_answers, activation_fn=None)

                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer)
                self.loss = tf.reduce_mean(self.cross_entropy)

                self.softmax = tf.nn.softmax(self.out, name='answer_prob')
                self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

                self.success = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))  # no need to compute the softmax

        with tf.variable_scope('accuracy'):
            self.accuracy = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            print('Model... CLEVR (baseline) build!')

    def get_loss(self):
        return self.loss

    def get_error(self):
        return 1 - self.accuracy

    def get_accuracy(self):
        return self.accuracy
