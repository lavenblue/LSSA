import numpy as np
import pandas as pd
import copy
import pickle


class data_generation():
    def __init__(self, type):
        print('init------------')
        self.data_type = type
        self.dataset = self.data_type + '/'+self.data_type + '_dataset.csv'

        self.train_users = []
        self.train_sessions = []      # 当前的session
        self.train_pre_sessions = []  # 之前的session集合
        self.train_long_neg = []      # 长期集合，随机采样得到的negative
        self.train_short_neg = []     # 短期集合，随机采样得到的negative

        self.test_users = []
        self.test_candidate_items = []
        self.test_sessions = []
        self.test_pre_sessions = []
        self.test_real_items = []

        self.user_number = 0
        self.item_number = 0

        self.gen_train_test_data()

        train = (self.train_users, self.train_pre_sessions, self.train_sessions,
                 self.train_long_neg, self.train_short_neg)

        test = (self.test_users, self.test_pre_sessions, self.test_sessions,
                self.test_real_items)

        pickle.dump(train, open(self.data_type + '/train.pkl', 'wb'))
        pickle.dump(test, open(self.data_type + '/test.pkl', 'wb'))

    def gen_train_test_data(self):
        self.data = pd.read_csv(self.dataset, names=['user', 'sessions'], dtype='str')
        is_first_line = 1
        maxLen_long = 0
        maxLen_short = 0
        for line in self.data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                self.user_purchased_item = dict()  # 保存每个用户购买记录，可用于train时负采样和test时剔除已打分商品
                is_first_line = 0
            else:
                user_id = int(line[0])
                sessions = [i for i in line[1].split('@')]
                size = len(sessions)
                the_first_session = [int(i)+1 for i in sessions[0].split(':')]
                tmp = copy.deepcopy(the_first_session)
                self.user_purchased_item[user_id] = tmp

                for j in range(1, size - 1):
                    # 每个用户的每个session在train_users中都对应着其user_id
                    self.train_users.append(user_id)
                    # test = sessions[j].split(':')
                    current_session = [int(it)+1 for it in sessions[j].split(':')]

                    self.user_purchased_item[user_id].extend(current_session)

                    self.train_sessions.append(current_session)

                    short_neg_items = []
                    for _ in range(len(current_session)-1):
                        short_neg_items.append(self.gen_neg(user_id))
                    self.train_short_neg.append(short_neg_items)

                    long_neg_items = []
                    for _ in range(len(self.user_purchased_item[user_id]) - 1):
                        long_neg_items.append(self.gen_neg(user_id))
                    self.train_long_neg.append(long_neg_items)
                    tmp = copy.deepcopy(self.user_purchased_item[user_id])
                    self.train_pre_sessions.append(tmp)
                    if len(current_session) > maxLen_short:
                        maxLen_short = len(current_session)

                # 对test的数据集也要格式化，test中每个用户都只有一个current session
                self.test_users.append(user_id)
                current_session = [int(it)+1 for it in sessions[size - 1].split(':')]
                item = current_session[-1]
                self.test_real_items.append(int(item))
                current_session.remove(item)
                self.test_sessions.append(current_session)

                self.user_purchased_item[user_id].extend(current_session)
                self.test_pre_sessions.append(self.user_purchased_item[user_id])

                if len(self.user_purchased_item[user_id]) > maxLen_long:
                    maxLen_long = len(self.user_purchased_item[user_id])

        print('maxLen_long = ', maxLen_long)
        print('maxLen_short = ', maxLen_short)

    def gen_neg(self, user_id):
        neg_item = np.random.randint(self.item_number)
        while neg_item in self.user_purchased_item[user_id]:
            neg_item = np.random.randint(self.item_number)
        return neg_item


if __name__ == '__main__':
    type = ['gowalla']
    dg = data_generation(type[0])
