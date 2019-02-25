from modules import *

class Model():
    def __init__(self, itemnum, args, maxlen, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())

        self.input_seq = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, maxlen))
        pos = self.pos
        neg = self.neg
        # mask是为了识别出序列中元素为0的位置，也就是当序列长度不足最大序列长度时补0的位置
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                # 位置编码
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            # emdedding + 位置编码
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            # for i in range(4):
            #     params = {"inputs": self.seq, "filters": args.hidden_units, "kernel_size": 5,
            #               "use_bias": True, "padding": "same"}
            #     outputs = tf.layers.conv1d(**params)
            # self.seq = tf.nn.relu(outputs)

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, args.hidden_units])

        self.test_logits = tf.matmul(seq_emb, tf.transpose(item_emb_table[1:]))
        # self.test_logits [1, n, 101]
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], maxlen, itemnum])
        # 取最后一行来预测下一个item
        self.test_logits = self.test_logits[:, -1, :]
        self.top_value, self.top_index = tf.nn.top_k(self.test_logits, k=itemnum, sorted=True)

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, seq):
        return sess.run(self.top_index,
                        {self.input_seq: seq, self.is_training: False})
