import tensorflow as tf
from tensorflow.contrib import layers


class SSNet(object):
    def __init__(self, config):
        self.config = config

    def add_placeholder(self):
        self.questions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.question_legnths = tf.placeholder(shape=[None], dtype=tf.int32)
        # [batch, num_words]
        self.input_words = tf.placeholder(shape=[None, None], dtype=tf.int32)
        # [batch, num_sentences, num_words]
        self.sentences = tf.placeholder(shape=[None, None, None], dtype=tf.int32)
        # [num_sentences, num_words]
        self.sequence_lengths = tf.placeholder(shape=[None, None], dtype=tf.int32)
        # [num_sentences]
        self.sentence_lengths = tf.placeholder(shape=[None], dtype=tf.int32)
        self.sentence_idx = tf.placeholder(shape=[None], dtype=tf.int32)
        self.answerable = tf.placeholder(shape=[None], dtype=tf.int32)
        self.answer_span = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.dropout = tf.placeholder(dtype=tf.float32)

        self.document_size, self.sentence_size, self.word_size = tf.unstack(tf.shape(self.sentences))

    def add_embeddings(self):
        self.embedding_matrix = tf.get_variable(shape=[self.config.vocab_size, self.config.embedding_size],
                                                initializer=layers.xavier_initializer(), name="embedding")
        self.embedded_questions = tf.nn.embedding_lookup(self.embedding_matrix, self.questions)
        self.embedded_words = tf.nn.embedding_lookup(self.embedding_matrix, self.input_words)
        self.embedded_sentences = tf.nn.embedding_lookup(self.embedding_matrix, self.sentences)

    def bi_lstm_embedding(self, inputs, sequence_length, return_last=False):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size)
        outputs, (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                          inputs, sequence_length,
                                                                          dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)
        last_states = tf.concat([fw_states[-1], bw_states[-1]], axis=1)
        if return_last:
            return last_states
        else:
            return outputs

    def add_word_lstm(self):
        with tf.variable_scope("bi-lstm-word"):
            # [b * s, w, d]
            reshaped_sentences = tf.reshape(self.embedded_sentences,
                                            [-1, self.word_size, self.config.embedding_size])
            sentence_lengths = tf.reshape(self.sentence_lengths, [-1])
            self.sentence_lstm = self.bi_lstm_embedding(reshaped_sentences, sentence_lengths)
        with tf.variable_scope("bi-lstm_question"):
            # get last hidden states and concatenate bi-directional hidden states
            # [batch, d * 2]
            self.question_lstm = self.bi_lstm_embedding(self.embedded_questions,
                                                        self.question_legnths,
                                                        return_last=True)

    def add_word_attention(self):
        # attend each sentence given the question
        # [b, 1, d * 2 ] -> [b * s, w, d * 2]
        with tf.variable_scope("word_attention"):
            query = tf.tile(tf.expand_dims(self.question_lstm, axis=1), [self.sentence_size, self.word_size, 1])
            attention_input = tf.concat([query, self.sentence_lstm], axis=2)
            # [b * s, w, attention_size]
            projected = layers.fully_connected(attention_input,
                                               self.config.attention_size,
                                               activation_fn=tf.nn.elu)
            v = tf.get_variable(shape=[self.config.attention_size, 1],
                                initializer=layers.xavier_initializer(),
                                name="v")
            reshaped_projected = tf.reshape(projected, [-1, self.config.attention_size])
            # [b * s, w, 1]
            attention_score = tf.matmul(reshaped_projected, v)
            attention_weight = tf.nn.softmax(attention_score, 1)
            sentence_vector = tf.reduce_sum(self.sentence_lstm * attention_weight, axis=1)
            self.sentence_vectors = tf.reshape(sentence_vector, [self.document_size, self.sentence_size, -1])

    def auxiliary_loss(self, attention_score, document_vector):
        attention_logits = tf.squeeze(attention_score, axis=-1)
        attention_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=attention_logits,
                                                                        labels=self.sentence_idx)
        attention_loss = tf.reduce_mean(attention_loss)
        binary_logits = layers.fully_connected(document_vector, 2, activation_fn=None)
        logistic_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=binary_logits,
                                                                       labels=self.answerable)
        return attention_loss, logistic_loss

    def add_sentence_attention(self):
        with tf.variable_scope("sentence_attention"):
            query = tf.tile(tf.expand_dims(self.question_lstm), [1, self.sentence_size, 1])
            document_lstm = self.bi_lstm_embedding(self.sentence_vectors,
                                                   self.sentence_lengths)
            attention_input = tf.concat([query, document_lstm], axis=2)
            projected = layers.fully_connected(attention_input,
                                               self.config.attention_size,
                                               activation_fn=tf.nn.elu)
            reshaped_projected = tf.reshape(projected, [-1, self.config.attention_size])
            v = tf.get_variable(shape=[self.config.attention_size, 1], name="v")
            attention_score = tf.matmul(reshaped_projected, v)
            attention_weight = tf.nn.softmax(attention_score, 1)
            self.document_vector = tf.reduce_sum(document_lstm * attention_weight, axis=1)
            self.attention_loss, self.logistic_loss = self.auxiliary_loss(attention_score,
                                                                          self.document_vector)

    # TODO : implement QA module, debug sentence selector, add optimizer
