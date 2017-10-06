import tensorflow as tf
from generic.tf_models.abstract_network import ResnetModel

from generic.tf_models import rnn
from generic.tf_models.image_feature import get_image_features
from neural_toolbox import utils


class CLEVRNetwork(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "clevr", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:

            self.batch_size = None

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size], name='answer')

            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep),
                                   lambda: tf.constant(1.0))

            word_emb = utils.get_embedding(self._question,
                                           n_words=no_words,
                                           n_dim=int(config["word_embedding_dim"]),
                                           scope="word_embedding")

            # word_emb = tf.nn.dropout(word_emb, dropout_keep)

            self.question_lstm, self.all_lstm_states = rnn.variable_length_LSTM(
                word_emb,
                num_hidden=int(config["no_hidden_LSTM"]),
                dropout_keep_prob=dropout_keep,
                seq_length=self._seq_length,
                depth=int(config["no_LSTM_cell"]),
                scope="question_lstm")

            #####################
            #   PICTURES
            #####################

            self._picture = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='picture')
            self.picture_out = get_image_features(
                    image=self._picture, question=self.question_lstm,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['image']
                )


            #####################
            #   COMBINE
            #####################
            activation_name = config["activation"]
            with tf.variable_scope('final_mlp'):

                self.question_embedding = utils.fully_connected(self.question_lstm, config["no_question_mlp"], activation=activation_name, scope='question_mlp')
                self.picture_embedding = utils.fully_connected(self.picture_out, config["no_picture_mlp"], activation=activation_name, scope='picture_mlp')

                full_embedding = self.picture_embedding * self.question_embedding
                full_embedding = tf.nn.dropout(full_embedding, dropout_keep)

                out = utils.fully_connected(full_embedding, config["no_hidden_final_mlp"], scope='hidden_final', activation=activation_name)
                out = tf.nn.dropout(out, dropout_keep)
                out = utils.fully_connected(out, no_answers, activation='linear', scope='layer_softmax')


            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=self._answer, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy)

            self.softmax = tf.nn.softmax(out, name='answer_prob')
            self.prediction = tf.argmax(out, axis=1, name='predicted_answer')  # no need to compute the softmax

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.prediction, self._answer)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')


