from modules import *


class Model():
    def __init__(self, itemnum, args, maxlen_long, maxlen_short, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.long_seq = tf.placeholder(tf.int32, shape=(None, maxlen_long))
        self.long_pos = tf.placeholder(tf.int32, shape=(None, maxlen_long))
        self.long_neg = tf.placeholder(tf.int32, shape=(None, maxlen_long))

        self.short_seq = tf.placeholder(tf.int32, shape=(None, maxlen_short))
        self.short_pos = tf.placeholder(tf.int32, shape=(None, maxlen_short))
        self.short_neg = tf.placeholder(tf.int32, shape=(None, maxlen_short))

        long_pos = self.long_pos
        long_neg = self.long_neg
        short_pos = self.short_pos
        short_neg = self.short_neg

        # mask是为了识别出序列中元素为0的位置，也就是当序列长度不足最大序列长度时补0的位置
        mask_long = tf.expand_dims(tf.to_float(tf.not_equal(self.long_seq, 0)), -1)
        mask_short = tf.expand_dims(tf.to_float(tf.not_equal(self.short_seq, 0)), -1)

        # sequence embedding, item embedding table
        self.long, item_emb_table = embedding(self.long_seq, vocab_size=itemnum + 1, num_units=args.hidden_units,
                                              zero_pad=True, scale=True, l2_reg=args.l2_emb,
                                              scope="input_embeddings", with_t=True, reuse=reuse)

        self.short = tf.nn.embedding_lookup(item_emb_table, self.short_seq)

        with tf.variable_scope("SASRec_long", reuse=reuse):

            # Dropout
            self.long = tf.layers.dropout(self.long, rate=args.dropout_rate,
                                          training=tf.convert_to_tensor(self.is_training))
            self.long *= mask_long

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.long = multihead_attention(queries=self.long, keys=self.long,
                                                    num_units=args.hidden_units, num_heads=args.num_lheads,
                                                    dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                    causality=True, scope="self_attention")

                    # Feed forward
                    self.long = feedforward(normalize(self.long), num_units=[args.hidden_units, args.hidden_units],
                                            dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.long *= mask_long

            self.long = normalize(self.long)

        with tf.variable_scope("SASRec_short", reuse=reuse):

            # Dropout
            self.short = tf.layers.dropout(self.short, rate=args.dropout_rate,
                                           training=tf.convert_to_tensor(self.is_training))
            self.short *= mask_short

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.short = multihead_attention(queries=self.short, keys=self.short,
                                                     num_units=args.hidden_units, num_heads=args.num_sheads,
                                                     dropout_rate=args.dropout_rate, is_training=self.is_training,
                                                     causality=True, scope="self_attention")

                    # Feed forward
                    self.short = feedforward(normalize(self.short), num_units=[args.hidden_units, args.hidden_units],
                                             dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.short *= mask_short

            self.short = normalize(self.short)
        print('self-long = ', self.long)
        print('self-short = ', self.short)

        long_pos = tf.reshape(long_pos, [tf.shape(self.long_seq)[0] * maxlen_long])
        long_neg = tf.reshape(long_neg, [tf.shape(self.long_seq)[0] * maxlen_long])
        lpos_emb = tf.nn.embedding_lookup(item_emb_table, long_pos)
        lneg_emb = tf.nn.embedding_lookup(item_emb_table, long_neg)

        short_pos = tf.reshape(short_pos, [tf.shape(self.short_seq)[0] * maxlen_short])
        short_neg = tf.reshape(short_neg, [tf.shape(self.short_seq)[0] * maxlen_short])
        spos_emb = tf.nn.embedding_lookup(item_emb_table, short_pos)
        sneg_emb = tf.nn.embedding_lookup(item_emb_table, short_neg)

        lseq_emb = tf.reshape(self.long, [tf.shape(self.long_seq)[0] * maxlen_long, args.hidden_units])
        sseq_emb = tf.reshape(self.short, [tf.shape(self.short_seq)[0] * maxlen_short, args.hidden_units])
        # 拼接方式1：+
        expand = tf.zeros([tf.shape(self.long_seq)[0], (maxlen_long - maxlen_short),
                           args.hidden_units], dtype=tf.float32)
        expand_semb = tf.concat([expand, self.short], axis=1)
        print('expand_current_emb = ', expand_semb)
        l_expand = tf.expand_dims(self.long, axis=-1)
        s_expand = tf.expand_dims(expand_semb, axis=-1)
        print('l_expand = ', l_expand)
        print('s_expand = ', s_expand)
        seq_emb = tf.reduce_sum(tf.concat([l_expand, s_expand], axis=-1), axis=-1)

        seq_emb = tf.reshape(seq_emb, [tf.shape(self.long_seq)[0] * maxlen_long, args.hidden_units])
        self.test_logits = tf.matmul(seq_emb, tf.transpose(item_emb_table[1:]))

        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.long_seq)[0], maxlen_long, itemnum])
        # 取最后一行来预测下一个item
        self.test_logits = self.test_logits[:, -1, :]
        self.top_value, self.top_index = tf.nn.top_k(self.test_logits, k=itemnum, sorted=True)

        # prediction layer
        self.lpos_logits = tf.reduce_sum(lpos_emb * lseq_emb, -1)
        self.lneg_logits = tf.reduce_sum(lneg_emb * lseq_emb, -1)
        self.spos_logits = tf.reduce_sum(spos_emb * sseq_emb, -1)
        self.sneg_logits = tf.reduce_sum(sneg_emb * sseq_emb, -1)

        # ignore padding items (0)
        istarget_long = tf.reshape(tf.to_float(tf.not_equal(long_pos, 0)), [tf.shape(self.long_seq)[0] * maxlen_long])
        istarget_short = tf.reshape(tf.to_float(tf.not_equal(short_pos, 0)), [tf.shape(self.short_seq)[0] * maxlen_short])

        self.loss = tf.reduce_mean(tf.log(1 + tf.exp(self.sneg_logits - self.spos_logits))) \
                    + tf.reduce_mean(tf.log(1 + tf.exp(self.lneg_logits - self.lpos_logits)))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.lpos_logits - self.lneg_logits) + 1) / 2) * istarget_long
        ) / tf.reduce_sum(istarget_long)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, long_seq, short_seq):
        return sess.run(self.top_index,
                        {self.long_seq: long_seq, self.short_seq: short_seq, self.is_training: False})
