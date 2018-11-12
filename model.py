import tensorflow as tf
from tensorflow.contrib import layers
import math


# TODO: GloVE, ELMO embedding
class SSQANet(object):
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

    def build_model(self):
        # add place holder
        self.contexts = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.context_legnths = tf.placeholder(shape=[None], dtype=tf.int32)
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
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

        self.document_size, self.sentence_size, self.word_size = tf.unstack(tf.shape(self.sentences))
        # add embeddings
        zeros = tf.constant([[0.0] * self.config.embedding_size])
        embedding_matrix = tf.get_variable(shape=[self.config.vocab_size - 1, self.config.embedding_size],
                                           initializer=layers.xavier_initializer(), name="embedding")

        self.embedding_matrix = tf.concat([zeros, embedding_matrix], axis=0)
        self.embedded_words = tf.nn.embedding_lookup(self.embedding_matrix, self.input_words)
        self.embedded_sentences = tf.nn.embedding_lookup(self.embedding_matrix, self.sentences)
        self.embedded_context = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
        self.embedded_questions = tf.nn.embedding_lookup(self.embedding_matrix, self.questions)

        # conv block and self attention block
        with tf.variable_scope("Embedding_Encoder_Layer"):
            contexts = self.residual_block(self.embedded_context, self.context_legnths,
                                           num_blocks=1, num_conv_blocks=4, kernel_size=7,
                                           num_filters=128, scope="Embedding_Encoder", reuse=False)
            questions = self.residual_block(self.embedded_questions, self.question_legnths, num_blocks=1,
                                            num_conv_blocks=4, kernel_size=7, num_filters=128,
                                            scope="Embedding_Encoder", reuse=True)

        with tf.variable_scope("hierarchical_attention"):
            # [b * s, w, d]
            reshaped_sentences = tf.reshape(self.embedded_sentences,
                                            [-1, self.word_size, self.config.embedding_size])
            sentence_lengths = tf.reshape(self.sequence_lengths, [-1])
            sentence_lstm = self.bi_lstm_embedding(reshaped_sentences, sentence_lengths,
                                                   scope="word_encoder", reuse=False)
            encoded_question = self.question_encoding(questions, self.question_legnths)
            sentence_vectors = self.word_level_attention(encoded_question, sentence_lstm, self.document_size,
                                                         self.sentence_size, self.word_size, self.sequence_lengths)
            document_vector, sentence_score = self.sentence_level_attention(encoded_question, sentence_vectors,
                                                                            self.sentence_size, self.sentence_lengths)

            self.attention_loss, self.binary_loss = self.auxiliary_loss(sentence_score,
                                                                        document_vector,
                                                                        encoded_question)
        with tf.variable_scope("Context_Query_Attention_Layer"):
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

        with tf.variable_scope("Output_Layer"):
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
            self.cross_entropy_loss = tf.reduce_mean(losses1 + losses2)
            self.loss = self.cross_entropy_loss \
                        + self.config.alpha * self.attention_loss \
                        + self.config.beta * self.binary_loss
        # for inference
        logits1 = tf.nn.softmax(start_logits)
        logits2 = tf.nn.softmax(end_logits)
        outer_product = tf.matmul(tf.expand_dims(logits1, axis=2), tf.expand_dims(logits2, axis=1))
        outer = tf.matrix_band_part(outer_product, 0, self.config.ans_limit)
        self.start = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.end = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
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
            outputs = None

            for i in range(num_conv_blocks):
                residual = inputs
                # apply layer normalization
                normalized = layers.layer_norm(inputs)
                if i % 2 == 0:
                    # apply dropout
                    normalized = tf.layers.dropout(normalized, self.dropout)
                outputs = self.depthwise_separable_conv(normalized, kernel_size, num_filters,
                                                        scope="depthwise_conv_{}".format(i), reuse=reuse)
                outputs = self.layer_dropout(outputs, residual, self.dropout * float(l) / L)
            return outputs, l

    def depthwise_separable_conv(self, inputs, kernel_size, num_filters, scope, reuse):
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
                                             strides=(1, 1, 1, 1), padding="SAME")
            outputs = tf.nn.relu(outputs + bias)
            # recover to the original shape [b, t, d]
            outputs = tf.squeeze(outputs, axis=2)
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
        # [b, n, m, 1] -> [b, n, m]
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
        # attention_score = self.quadlinear_attention(questions, contexts, document_vector)
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

    def bi_lstm_embedding(self, inputs, sequence_length, scope, reuse, return_last=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size)
            outputs, (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                              inputs, sequence_length,
                                                                              dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.layers.dropout(outputs, self.dropout)
            last_states = tf.concat([fw_states[-1], bw_states[-1]], axis=1)
            if return_last:
                return last_states
            else:
                return outputs

    def word_level_attention(self, question_lstm, sentence_lstm, document_size,
                             sentence_size, word_size, sequence_lengths):
        # attend each sentence given the question
        # [b, 1, d * 2 ] -> [b * s, w, d * 2]
        with tf.variable_scope("word_attention"):
            query = tf.tile(tf.expand_dims(question_lstm, axis=1),
                            [sentence_size, word_size, 1])
            attention_input = tf.concat([query, sentence_lstm], axis=2)
            # [b * s, w, attention_size]
            projected = layers.fully_connected(attention_input,
                                               self.config.attention_size,
                                               weights_initializer=layers.variance_scaling_initializer(),
                                               weights_regularizer=self.regularizer,
                                               activation_fn=tf.nn.elu)
            v = tf.get_variable(shape=[self.config.attention_size, 1],
                                initializer=layers.xavier_initializer(),
                                regularizer=self.regularizer,
                                name="v")
            # [b * s, w , d] -> [b * s * w , d]
            reshaped_projected = tf.reshape(projected, [-1, self.config.attention_size])
            # [b * s * w, 1]
            attention_score = tf.matmul(reshaped_projected, v)
            # reshape to original shape [b*s*w, 1] -> [b*s, w, 1]
            attention_score = tf.reshape(attention_score, [-1, word_size, 1])

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
                                           self.config.lstm_size * 2])
            return sentence_vectors

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
        self.preds = tf.argmax(binary_logits, axis=1, output_type=tf.int32)
        correct_pred = tf.equal(self.preds, self.answerable)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
        logistic_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=binary_logits,
                                                                       labels=self.answerable)
        logistic_loss = tf.reduce_mean(logistic_loss)

        return attention_loss, logistic_loss

    def sentence_level_attention(self, question_vector, sentence_vectors, sentence_size, sentence_lengths):
        with tf.variable_scope("sentence_attention") as scope:
            query = tf.tile(tf.expand_dims(question_vector, 1), [1, self.sentence_size, 1])
            # [b, s, 2 * d]
            document_lstm = self.bi_lstm_embedding(sentence_vectors,
                                                   sentence_lengths,
                                                   scope, reuse=False)
            attention_input = tf.concat([query, document_lstm], axis=2)
            projected = layers.fully_connected(attention_input,
                                               self.config.attention_size,
                                               weights_initializer=layers.xavier_initializer(),
                                               weights_regularizer=self.regularizer,
                                               activation_fn=tf.nn.elu)
            # [b ,s, d] -> [b * s, d]
            reshaped_projected = tf.reshape(projected, [-1, self.config.attention_size])
            v = tf.get_variable(shape=[self.config.attention_size, 1], name="v", regularizer=self.regularizer)
            attention_score = tf.matmul(reshaped_projected, v)
            # [b * s, 1] -> [b, s, 1]
            attention_score = tf.reshape(attention_score, [-1, sentence_size, 1])

            # -inf score for zero padding
            mask = tf.sequence_mask(sentence_lengths, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            padding = tf.ones_like(attention_score) * (-2 ** 32 + 1)
            attention_score = tf.where(tf.equal(mask, 0), padding, attention_score)

            attention_weight = tf.nn.softmax(attention_score, 1)
            document_vector = tf.reduce_sum(document_lstm * attention_weight, axis=1)
            return document_vector, attention_score

    def add_train_op(self):
        with tf.variable_scope("adam_opt"):
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
        output_feed = [self.train_op, self.loss, self.acc, self.preds]
        _, loss, acc, pred = self.sess.run(output_feed, feed_dict)
        return loss, acc, pred

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
            self.dropout: 1.0
        }
        output_feed = [self.start, self.end]
        start, end = self.sess.run(output_feed, feed_dict)
        return start, end
