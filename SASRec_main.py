import time
import argparse
import tensorflow as tf
from SASRec_model import Model
from util import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gowalla')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=100, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    dataset = sas_partition('data/'+args.dataset+'/'+args.dataset+'_dataset.csv')
    [user_train, user_test, usernum, itemnum] = dataset

    model = Model(itemnum, args, args.maxlen)
    sess.run(tf.global_variables_initializer())

    T = 0.0
    t0 = time.time()

    inputs_user, inputs_seq, inputs_pos, inputs_neg = generate_train_data(user_train, itemnum, args.maxlen)
    test_user, test_seqs, test_item = generate_test_data(user_train, user_test, usernum, args.maxlen)

    num_batch = int(len(user_train) / args.batch_size)

    for epoch in range(1, args.num_epochs + 1):
        print('training--------------', epoch)
        for b in range(num_batch + 1):
            if b < num_batch:
                low = b * args.batch_size
                high = (b + 1) * args.batch_size
            else:
                low = b * args.batch_size
                high = len(inputs_seq)
            u = inputs_user[low: high]
            seq = inputs_seq[low: high]
            pos = inputs_pos[low: high]
            neg = inputs_neg[low: high]

            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

        if epoch >= 5:
            t1 = time.time() - t0
            T += t1
            print('Evaluating-------------', epoch)
            top_K = [1, 5, 10, 15, 20]
            pre_result = {}
            rec_result = {}
            ndcg_result = {}
            apks = []
            for k in top_K:
                pre_result[k] = []
                rec_result[k] = []
                ndcg_result[k] = []

            for user_id in range(len(test_user)):
                seq = test_seqs[user_id]
                real_item = [test_item[user_id]-1]

                top_index = model.predict(sess, [seq])
                apks.append(compute_apk(real_item, top_index[0], k=np.inf))

                for k in top_K:
                    prec, rec, ndcg = precision_k(top_index[0], real_item, k)
                    pre_result[k].append(prec)
                    rec_result[k].append(rec)
                    ndcg_result[k].append(ndcg)

            for k in top_K:
                print('precision@' + str(k) + ' = ' + str(np.mean(pre_result[k])) + ',  ' +
                      'recall@' + str(k) + ' = ' + str(np.mean(rec_result[k])) + ',  ' +
                      'ndcg@' + str(k) + ' = ' + str(np.mean(ndcg_result[k])))
            print('MAP' + ' = ' + str(np.mean(apks)))

            t0 = time.time()

    print("Done")
