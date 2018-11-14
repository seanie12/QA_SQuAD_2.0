import tensorflow as tf
from tensorflow.contrib import layers
import math
import os


# TODO: ELMO embedding
class SSQANet(object):
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.merged = None
        self.summary_writer = None

    def build_model(self):
        # add place holder
        self.contexts = tf.placeholder(shape=[None, None], dtype=tf.int32, name="context")
        self.context_legnths = tf.placeholder(shape=[None], dtype=tf.int32, name="c_length")
        self.questions = tf.placeholder(shape=[None, None], dtype=tf.int32, name="q")
        self.question_legnths = tf.placeholder(shape=[None], dtype=tf.int32, name="q_len")
        # [batch, num_sentences, num_words]
        self.sentences = tf.placeholder(shape=[None, None, None], dtype=tf.int32, name="sentences")
        # [num_sentences, num_words]
        self.sequence_lengths = tf.placeholder(shape=[None, None], dtype=tf.int32, name="seq_len")
        # [num_sentences]
        self.sentence_lengths = tf.placeholder(shape=[None], dtype=tf.int32, name="sent_len")
        self.sentence_idx = tf.placeholder(shape=[None], dtype=tf.int32, name="sent_idx")
        self.answerable = tf.placeholder(shape=[None], dtype=tf.int32, name="answ")
        self.answer_span = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="answer_span")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

        self.avg_loss = tf.placeholder(dtype=tf.float32, name="avg_loss")
        self.avg_em = tf.placeholder(dtype=tf.float32, name="avg_em")
        self.avg_acc = tf.placeholder(dtype=tf.float32, name="avg_acc")
        loss_summary = tf.summary.scalar("loss", self.avg_em)
        acc_summary = tf.summary.scalar("accuracy", self.avg_acc)
        em_summary = tf.summary.scalar("em", self.avg_em)
        self.merged = tf.summary.merge([loss_summary, acc_summary, em_summary])

        self.document_size, self.sentence_size, self.word_size = tf.unstack(tf.shape(self.sentences))
        # add embeddings
        zeros = tf.constant([[0.0] * self.config.embedding_size])
        unk_dummy = tf.get_variable(shape=[2, self.config.embedding_size],
                                    initializer=layers.xavier_initializer(), name="special_token")
        # load pre-trained GloVe
        embedding_matrix = tf.Variable(initial_value=self.config.embeddings, trainable=False,
                                       dtype=tf.float32, name="embedding")
        self.embedding_matrix = tf.concat([zeros, unk_dummy, embedding_matrix], axis=0)
        self.embedded_sentences = tf.nn.embedding_lookup(self.embedding_matrix, self.sentences)
        self.embedded_sentences = tf.layers.dropout(self.embedded_sentences, self.dropout)
        self.embedded_context = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
        self.embedded_context = tf.layers.dropout(self.embedded_context, self.dropout)
        self.embedded_questions = tf.nn.embedding_lookup(self.embedding_matrix, self.questions)
        self.embedded_questions = tf.layers.dropout(self.embedded_questions, self.dropout)
        # conv block and self attention block
        with tf.variable_scope("Embedding_Encoder_Layer"):
            contexts = self.residual_block(self.embedded_context, self.context_legnths,
                                           num_blocks=1, num_conv_blocks=4, kernel_size=7,
                                           num_filters=128, scope="Embedding_Encoder", reuse=False)
            questions = self.residual_block(self.embedded_questions, self.question_legnths, num_blocks=1,
                                            num_conv_blocks=4, kernel_size=7, num_filters=128,
                                            scope="Embedding_Encoder", reuse=True)
            reshaped_sentences = tf.reshape(self.embedded_sentences,
                                            [-1, self.word_size, self.config.embedding_size])
            sentence_len = tf.reshape(self.sequence_lengths, [-1])
            encoded_sentence = self.residual_block(reshaped_sentences, sentence_len, num_blocks=1,
                                                   num_conv_blocks=1, kernel_size=7,
                                                   num_filters=128, scope="Embedding_Encoder", reuse=True)

        with tf.variable_scope("hierarchical_attention") and tf.device("/device:GPU:0"):
            # [b * s, w, d]
            cnn_inputs = tf.layers.dense(encoded_sentence, self.config.filter_size,
                                         kernel_regularizer=self.regularizer,
                                         kernel_initializer=layers.xavier_initializer(),
                                         activation=tf.nn.relu)
            sentence_cnn = self.conv_encoder(cnn_inputs, self.config.filter_size,
                                             scope="word_encoder", reuse=False)
            encoded_question = self.question_encoding(questions, self.question_legnths)
            # [b, s, d]
            sentence_vectors = self.word_level_attention(encoded_question, sentence_cnn, self.document_size,
                                                         self.sentence_size, self.word_size, self.sequence_lengths)
            sentence_cnn = self.conv_encoder(sentence_vectors, self.config.filter_size,
                                             scope="sentence_encoder", reuse=False)
            document_vector, sentence_score = self.sentence_level_attention(encoded_question, sentence_cnn,
                                                                            self.sentence_size, self.sentence_lengths)

            self.attention_loss, self.binary_loss = self.auxiliary_loss(sentence_score,
                                                                        document_vector,
                                                                        encoded_question)
        with tf.variable_scope("Context_Query_Attention_Layer") and tf.device("/device:GPU:0"):
            A, B = self.co_attention(questions, contexts,
                                     self.question_legnths, self.context_legnths)
            attention_outputs = [contexts, A, contexts * A, contexts * B]
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=2)
            inputs = tf.layers.dense(inputs, self.config.attention_size,
                                     kernel_regularizer=self.regularizer,
                                     kernel_initializer=layers.variance_scaling_initializer(),
                                     activation=tf.nn.relu)
            memories = []
            for i in range(3):
                outputs = self.residual_block(inputs, self.context_legnths,
                                              num_blocks=7, num_conv_blocks=2,
                                              num_filters=128, kernel_size=5,
                                              scope="Model_Encoder",
                                              reuse=True if i > 0 else False)
                if i == 2:
                    outputs = tf.layers.dropout(outputs, self.dropout)
                memories.append(outputs)
                inputs = outputs

        with tf.variable_scope("Output_Layer") and tf.device("/device:GPU:0"):
            logits_inputs = tf.concat([memories[0], memories[1]], axis=2)
            start_logits = self.pointer_network(document_vector, logits_inputs,
                                                self.context_legnths, scope="start_logits")
            logits_inputs = tf.concat([memories[0], memories[2]], axis=2)
            end_logits = self.pointer_network(document_vector, logits_inputs,
                                              self.context_legnths, scope="end_logits")

            start_label, end_label = tf.split(self.answer_span, 2, axis=1)
            start_label = tf.squeeze(start_label, axis=-1)
            end_label = tf.squeeze(end_label, axis=-1)
            losses1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logits, labels=start_label)
            losses2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logits, labels=end_label)
            cross_entropy_loss = tf.reduce_mean(losses1 + losses2)
            self.loss = cross_entropy_loss \
                        + self.config.alpha * self.attention_loss \
                        + self.config.beta * self.binary_loss

        # for inference
        logits1 = tf.nn.softmax(start_logits)
        logits2 = tf.nn.softmax(end_logits)
        outer_product = tf.matmul(tf.expand_dims(logits1, axis=2), tf.expand_dims(logits2, axis=1))
        outer = tf.matrix_band_part(outer_product, 0, self.config.ans_limit)
        self.start = tf.argmax(tf.reduce_max(outer, axis=2), axis=1, output_type=tf.int32)
        self.end = tf.argmax(tf.reduce_max(outer, axis=1), axis=1, output_type=tf.int32)
        self.em = self.evaluate_em(self.start, self.end, self.answer_span, self.unans_prob)
        if self.config.l2_lambda > 0:
            vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = layers.apply_regularization(self.regularizer, vars)
            self.loss += l2_loss
        # Exponential moving average
        self.var_ema = tf.train.ExponentialMovingAverage(0.9999)
        ema_op = self.var_ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

            self.assign_vars = []
            for var in tf.global_variables():
                v = self.var_ema.average(var)
                if v:
                    self.assign_vars.append(tf.assign(var, v))

        self.add_train_op()
        self.init_session()

    def evaluate_em(self, start_preds, end_preds, answer_spans, unans_probs, threshold=0.5):
        # if the question is unanswerable, answer span should be last token idx
        dummy_idx = self.context_legnths - 1
        start_preds = tf.where(tf.greater_equal(unans_probs, threshold), dummy_idx, start_preds)
        end_preds = tf.where(tf.greater_equal(unans_probs, threshold), dummy_idx, end_preds)
        ans_start, ans_end = tf.split(answer_spans, 2, axis=1)
        ans_start = tf.squeeze(ans_start, axis=-1)
        ans_end = tf.squeeze(ans_end, axis=-1)
        correct_start = tf.equal(start_preds, ans_start)
        correct_end = tf.equal(end_preds, ans_end)
        total_correct = tf.cast(tf.logical_and(correct_start, correct_end), dtype=tf.float32)
        em = tf.reduce_mean(total_correct)
        return em

    def pointer_network(self, query, keys, sequence_lengths, scope, reuse=False):
        # query : [b,d], keys: [b,t,d]
        with tf.variable_scope(scope, reuse=reuse):
            units = query.shape[-1]
            trans = tf.layers.dense(keys, units,
                                    activation=None, use_bias=False)
            q = tf.expand_dims(query, axis=-1)
            # [b,t,1] -> [b,t]
            attention = tf.squeeze(tf.matmul(trans, q), axis=-1)
            masked = self.mask_logits(attention, sequence_lengths)
        return masked

    def mask_logits(self, logits, sequence_lengths):
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        mask_value = -1e32
        return logits + mask_value * (1 - mask)

    def question_encoding(self, inputs, sequence_lengths):
        # [b, m, d] -> [b, m, 1]
        alpha = tf.layers.dense(inputs, 1, activation=None, use_bias=False,
                                kernel_initializer=layers.variance_scaling_initializer(),
                                kernel_regularizer=self.regularizer)
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        paddings = tf.ones_like(alpha) * (-2 ** 32 + 1)
        alpha = tf.where(tf.equal(mask, 0), paddings, alpha)
        alpha = tf.nn.softmax(alpha, 1)
        encoding = tf.reduce_sum(alpha * inputs, axis=1)
        return encoding

    @staticmethod
    def position_embeddings(inputs, sequence_length):
        length = tf.shape(inputs)[1]
        channels = tf.shape(inputs)[2]
        max_timescale = 1.0e4
        min_timescale = 1.0
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = channels // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        # mask for zero padding
        mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        signal *= mask
        return signal

    def layer_dropout(self, inputs, residual, dropout):
        cond = tf.random_uniform([]) < dropout
        return tf.cond(cond, lambda: residual, lambda: tf.layers.dropout(inputs, self.dropout) + residual)

    def residual_block(self, inputs, sequence_length, num_blocks, num_conv_blocks,
                       kernel_size, num_filters, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            sublayer = 1
            # conv_block * # + self attetion + feed forward
            total_sublayers = (num_conv_blocks + 2) * num_blocks
            dim = inputs.shape[-1]
            if dim != num_filters:
                inputs = tf.layers.dense(inputs, num_filters, activation=tf.nn.relu,
                                         kernel_regularizer=self.regularizer,
                                         kernel_initializer=layers.variance_scaling_initializer())
            for i in range(num_blocks):
                # add positional embedding
                inputs = inputs + self.position_embeddings(inputs, sequence_length)
                outputs, sublayer = self.conv_blocks(inputs, num_conv_blocks, kernel_size, num_filters,
                                                     scope="conv_block_{}".format(i), reuse=reuse,
                                                     sublayers=(sublayer, total_sublayers))
                outputs, sublayer = self.self_attention_block(outputs, sequence_length, (sublayer, total_sublayers),
                                                              scope="attention_block_{}".format(i), reuse=reuse)
                inputs = outputs
            return outputs

    def conv_blocks(self, inputs, num_conv_blocks, kernel_size,
                    num_filters, scope, reuse, sublayers=(1, 1)):
        with tf.variable_scope(scope, reuse=reuse):
            l, L = sublayers
            outputs = inputs

            for i in range(num_conv_blocks):
                residual = outputs
                # apply layer normalization
                normalized = layers.layer_norm(outputs)
                if i % 2 == 0:
                    # apply dropout
                    normalized = tf.layers.dropout(normalized, self.dropout)
                outputs = self.depthwise_separable_conv(normalized, kernel_size, num_filters,
                                                        scope="depthwise_conv_{}".format(i), reuse=reuse)
                outputs = self.layer_dropout(outputs, residual, self.dropout * float(l) / L)
            return outputs, l

    def depthwise_separable_conv(self, inputs, kernel_size, num_filters, scope, reuse, rate=(1, 1)):
        with tf.variable_scope(scope, reuse=reuse):
            # [batch, t, 1, d]
            inputs = tf.expand_dims(inputs, axis=2)
            depthwise_filter = tf.get_variable(shape=[kernel_size, 1, num_filters, 1],
                                               initializer=layers.variance_scaling_initializer(),
                                               name="depthwise_filter",
                                               regularizer=self.regularizer)
            pointwise_filter = tf.get_variable(shape=[1, 1, num_filters, num_filters],
                                               name="pointwise_filter",
                                               initializer=layers.variance_scaling_initializer(),
                                               regularizer=self.regularizer)
            bias = tf.get_variable(shape=[num_filters],
                                   initializer=tf.zeros_initializer(),
                                   name="bias",
                                   regularizer=self.regularizer)

            outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter,
                                             rate=rate, strides=(1, 1, 1, 1), padding="SAME")
            outputs = tf.nn.relu(outputs + bias)
            # recover to the original shape [b, t, d]
            outputs = tf.squeeze(outputs, axis=2)
            return outputs

    def conv_encoder(self, inputs, num_filters, scope, reuse):
        outputs = inputs
        with tf.variable_scope(scope, reuse=reuse):
            rates = [(1, 1), (2, 2), (4, 4)]
            for i, rate in enumerate(rates):
                residual = outputs
                outputs = self.depthwise_separable_conv(outputs, 5, num_filters,
                                                        scope="conv_encoder_{}".format(i),
                                                        reuse=False, rate=rate)
                outputs = layers.layer_norm(outputs)
                if i % 2 == 0:
                    outputs = tf.layers.dropout(outputs, self.dropout)
                outputs = self.layer_dropout(outputs, residual,
                                             self.dropout * float(i) * float(len(rates)))
            return outputs

    def self_attention_block(self, inputs, sequence_length, sublayers, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            l, L = sublayers
            inputs = layers.layer_norm(inputs)
            inputs = tf.layers.dropout(inputs, self.dropout)
            outputs = self.multihead_attention(inputs, sequence_length)
            outputs = self.layer_dropout(outputs, inputs, self.dropout * l / float(L))
            l += 1
            # FFN
            residual = layers.layer_norm(outputs)
            outputs = tf.layers.dropout(outputs, self.dropout)
            hiddens = tf.layers.dense(outputs, self.config.attention_size * 2,
                                      activation=tf.nn.elu)
            fc_outputs = tf.layers.dense(hiddens, self.config.attention_size,
                                         activation=None)
            outputs = self.layer_dropout(residual, fc_outputs, self.dropout * l / float(L))

        return outputs, l

    def multihead_attention(self, queries, sequence_length):
        Q = tf.layers.dense(queries, self.config.attention_size,
                            kernel_initializer=layers.variance_scaling_initializer(),
                            activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        K = tf.layers.dense(queries, self.config.attention_size,
                            kernel_initializer=layers.variance_scaling_initializer(),
                            activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        V = tf.layers.dense(queries, self.config.attention_size,
                            kernel_initializer=layers.variance_scaling_initializer(),
                            activation=tf.nn.relu, kernel_regularizer=self.regularizer)

        Q_ = tf.concat(tf.split(Q, self.config.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.config.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.config.num_heads, axis=2), axis=0)
        # attention weight and scaling
        weight = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        weight /= (self.config.attention_size // self.config.num_heads) ** 0.5
        # key masking : assign -inf to zero padding
        key_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        key_mask = tf.tile(key_mask, [self.config.num_heads, 1])
        key_mask = tf.tile(tf.expand_dims(key_mask, axis=1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(weight) * (-2 ** 32 + 1)
        weight = tf.where(tf.equal(key_mask, 0), paddings, weight)
        weight = tf.nn.softmax(weight)

        # query masking - assign zero to where query is zero padding token
        query_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
        query_mask = tf.tile(query_mask, [self.config.num_heads, 1])
        query_mask = tf.expand_dims(query_mask, axis=2)
        weight *= query_mask
        weight = tf.layers.dropout(weight, self.dropout)
        outputs = tf.matmul(weight, V_)
        outputs = tf.concat(tf.split(outputs, self.config.num_heads, axis=0), axis=2)

        return outputs

    def quadlinear_attention(self, questions, contexts, document_vector):
        # f(q,c) = W[q,c, q*c]
        # Q : [b, m, d] -> [b, n, m, d]
        # C : [b, n, d] -> [b, n, m, d]
        # D : [b,d]
        m = tf.shape(questions)[1]
        n = tf.shape(contexts)[1]
        questions = tf.tile(tf.expand_dims(questions, axis=1), [1, n, 1, 1])
        contexts = tf.tile(tf.expand_dims(contexts, axis=2), [1, 1, m, 1])
        docs = tf.expand_dims(tf.expand_dims(document_vector, axis=1), 1)
        docs = tf.tile(docs, [1, n, m, 1])
        quad = tf.concat([questions, contexts, questions * contexts, docs], axis=-1)
        # [b, n, m, 1] -> [b, n, m]
        score = tf.layers.dense(quad, 1, activation=None,
                                use_bias=False, kernel_regularizer=self.regularizer)
        score = tf.squeeze(score, axis=-1)
        return score

    def trilinear_attention(self, questions, contexts):
        # f(q,c) = W[q,c, q*c]
        # Q : [b, m, d] -> [b, n, m, d]
        # C : [b, n, d] -> [b, n, m, d]
        # D : [b,d]
        m = tf.shape(questions)[1]
        n = tf.shape(contexts)[1]
        questions = tf.tile(tf.expand_dims(questions, axis=1), [1, n, 1, 1])
        contexts = tf.tile(tf.expand_dims(contexts, axis=2), [1, 1, m, 1])
        tri = tf.concat([questions, contexts, questions * contexts], axis=-1)
        # [b, n, m, 3d] -> [b, n, m, 1] -> [b, n, m]
        score = tf.layers.dense(tri, 1, activation=None,
                                use_bias=False, kernel_regularizer=self.regularizer)
        score = tf.squeeze(score, axis=-1)
        return score

    def co_attention(self, questions, contexts, questions_lengths, contexts_lengths):
        # context to query attention
        # Q :[b, m, d], C :[b, n, d], D : [b, d]
        # S : [b, n, m]
        n = tf.shape(contexts)[1]
        m = tf.shape(questions)[1]
        attention_score = self.trilinear_attention(questions, contexts)
        S = attention_score
        # key masking : [b, m]
        key_masks = tf.sequence_mask(questions_lengths, dtype=tf.float32)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, n, 1])
        paddings = tf.ones_like(S) * (-2 ** 32 + 1)
        S = tf.where(tf.equal(key_masks, 0), paddings, S)
        S = tf.nn.softmax(S)
        # query_mask
        query_masks = tf.sequence_mask(contexts_lengths, dtype=tf.float32)
        query_masks = tf.expand_dims(query_masks, 2)
        S *= query_masks
        # S :[b, n, m], Q: [b, m, d], A :[b,n,d]
        A = tf.matmul(S, questions)

        S_ = tf.transpose(attention_score, [0, 2, 1])
        # key masks
        key_masks = tf.sequence_mask(contexts_lengths, dtype=tf.float32)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, m, 1])

        paddings = tf.ones_like(S_) * (-2 ** 32 + 1)
        S_ = tf.where(tf.equal(key_masks, 0), paddings, S_)
        S_ = tf.nn.softmax(S_)

        query_masks = tf.sequence_mask(questions_lengths, dtype=tf.float32)
        query_masks = tf.expand_dims(query_masks, 2)
        S_ *= query_masks

        q2c = tf.matmul(S_, contexts)
        B = tf.matmul(S, q2c)

        return A, B

    def word_level_attention(self, question_lstm, sentence_lstm, document_size,
                             sentence_size, word_size, sequence_lengths):
        # attend each sentence given the question
        # [b, 1, d] -> [b * s, w, d]
        with tf.variable_scope("word_attention"):
            query = tf.tile(tf.expand_dims(question_lstm, axis=1),
                            [sentence_size, word_size, 1])
            attention_input = tf.concat([query, sentence_lstm], axis=2)
            # [b * s, w, attention_size]
            projected = tf.layers.dense(attention_input,
                                        self.config.filter_size,
                                        kernel_initializer=layers.variance_scaling_initializer(),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.elu)
            projected = tf.layers.dropout(projected, self.dropout)

            # reshape to original shape [b*s, w, d] -> [b*s, w, 1]
            attention_score = tf.layers.dense(projected, 1,
                                              use_bias=False,
                                              kernel_initializer=layers.xavier_initializer(),
                                              kernel_regularizer=self.regularizer,
                                              activation=None)

            # -inf weight to zero padding
            sequence_length = tf.reshape(sequence_lengths, [-1])
            mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            padding = tf.ones_like(attention_score) * (-2 ** 32 + 1)
            attention_score = tf.where(tf.equal(mask, 0), padding, attention_score)

            attention_weight = tf.nn.softmax(attention_score, 1)
            sentence_vector = tf.reduce_sum(sentence_lstm * attention_weight, axis=1)
            sentence_vectors = tf.reshape(sentence_vector,
                                          [document_size, sentence_size,
                                           self.config.filter_size])
            return sentence_vectors

    def sentence_level_attention(self, question_vector, sentence_vectors, sentence_size, sentence_lengths):
        with tf.variable_scope("sentence_attention"):
            # [b, d] -> [b, s, d]
            query = tf.tile(tf.expand_dims(question_vector, 1), [1, sentence_size, 1])
            attention_input = tf.concat([query, sentence_vectors], axis=2)
            projected = tf.layers.dense(attention_input,
                                        self.config.attention_size,
                                        kernel_initializer=layers.xavier_initializer(),
                                        kernel_regularizer=self.regularizer,
                                        activation=tf.nn.elu)
            projected = tf.layers.dropout(projected, self.dropout)
            # [b * s, 1] -> [b, s, 1]
            attention_score = tf.layers.dense(projected, 1, use_bias=False,
                                              kernel_initializer=layers.xavier_initializer(),
                                              kernel_regularizer=self.regularizer,
                                              activation=None)

            # -inf score for zero padding
            mask = tf.sequence_mask(sentence_lengths, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            padding = tf.ones_like(attention_score) * (-2 ** 32 + 1)
            attention_score = tf.where(tf.equal(mask, 0), padding, attention_score)

            attention_weight = tf.nn.softmax(attention_score, 1)
            document_vector = tf.reduce_sum(sentence_vectors * attention_weight, axis=1)
            return document_vector, attention_score

    def auxiliary_loss(self, attention_score, document_vector, question_vector):
        # [b * s ,1] -> [b, s]
        attention_logits = tf.squeeze(attention_score, axis=-1)
        attention_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=attention_logits,
                                                                        labels=self.sentence_idx)
        attention_loss = tf.reduce_mean(attention_loss)
        logits_input = tf.concat([document_vector, question_vector], axis=-1)
        binary_logits = tf.layers.dense(logits_input, 2,
                                        kernel_initializer=layers.xavier_initializer(),
                                        kernel_regularizer=self.regularizer,
                                        use_bias=True,
                                        activation=None)
        self.unans_prob = binary_logits[:, 0]
        self.preds = tf.argmax(binary_logits, axis=1, output_type=tf.int32)
        correct_pred = tf.equal(self.preds, self.answerable)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
        logistic_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=binary_logits,
                                                                       labels=self.answerable)
        logistic_loss = tf.reduce_mean(logistic_loss)

        return attention_loss, logistic_loss

    def add_train_op(self):
        with tf.device("device:GPU:1"):
            lr = tf.minimum(self.config.lr,
                            0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
            opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
            vars = tf.trainable_variables()
            grads = tf.gradients(self.loss, vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            self.train_op = opt.apply_gradients(
                zip(clipped_grads, vars), global_step=self.global_step)

    def init_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter("./results/train", self.sess.graph)
        self.dev_writer = tf.summary.FileWriter("./results/dev", self.sess.graph)

    def save_session(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path)

    def restore_session(self, path):
        self.saver.restore(self.sess, path)

    def train(self, questions, question_length, contexts, context_lengths, sentences, sequence_length,
              sentence_length, sentence_idx, answerable, spans, dropout=1.0):
        feed_dict = {
            self.questions: questions,
            self.question_legnths: question_length,
            self.contexts: contexts,
            self.context_legnths: context_lengths,
            self.sentences: sentences,
            self.sentence_lengths: sentence_length,
            self.sequence_lengths: sequence_length,
            self.sentence_idx: sentence_idx,
            self.answerable: answerable,
            self.answer_span: spans,
            self.dropout: dropout
        }
        output_feed = [self.train_op, self.loss, self.acc, self.preds, self.global_step]
        _, loss, acc, pred, step = self.sess.run(output_feed, feed_dict)
        return loss, acc, pred, step

    def eval(self, questions, question_length, contexts,
             context_lengths, sentences, sequence_lengths,
             sentence_lengths, sentence_idx, answerable, span):
        feed_dict = {
            self.questions: questions,
            self.question_legnths: question_length,
            self.contexts: contexts,
            self.context_legnths: context_lengths,
            self.sentences: sentences,
            self.sequence_lengths: sequence_lengths,
            self.sentence_lengths: sentence_lengths,
            self.sentence_idx:sentence_idx,
            self.answerable: answerable,
            self.answer_span: span,
            self.dropout: 0.0
        }
        output_feed = [self.acc, self.em, self.loss]
        acc, em, loss = self.sess.run(output_feed, feed_dict)
        return acc, em, loss

    def write_summary(self, avg_acc, avg_em, avg_loss, mode):
        feed_dict = {
            self.avg_acc: avg_acc,
            self.avg_em: avg_em,
            self.avg_loss: avg_loss
        }
        summary, step = self.sess.run([self.merged, self.global_step], feed_dict)
        if mode == "train":
            self.train_writer.add_summary(summary, step)
        else:
            self.dev_writer.add_summary(summary, step)

    def infer(self, questions, question_length, contexts,
              context_lengths, sentences, sequence_lengths, sentence_lengths):
        feed_dict = {
            self.questions: questions,
            self.question_legnths: question_length,
            self.contexts: contexts,
            self.context_legnths: context_lengths,
            self.sentences: sentences,
            self.sequence_lengths: sequence_lengths,
            self.sentence_lengths: sentence_lengths,
            self.dropout: 0.0
        }
        output_feed = [self.start, self.end]
        start, end = self.sess.run(output_feed, feed_dict)
        return start, end
