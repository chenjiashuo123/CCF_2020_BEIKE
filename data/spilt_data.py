import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import os


'''数据划分'''
train_df=pd.read_csv("./process_data/train.tsv", sep='\t', header=0, encoding="utf-8")
train_df.columns=['q_id', 'r_id', 'query', 'reply', 'label']
test_df=pd.read_csv("./process_data/test.tsv", sep='\t', header=None, encoding="utf-8")
def generate_data(is_pse_label=True):
    skf = GroupKFold(n_splits=5)
    i = 0
    for train_index, dev_index in skf.split(train_df, groups=train_df.q_id):
        print(i, "TRAIN:", train_index, "TEST:", dev_index)
        DATA_DIR = "./data_KFold/data_origin_{}/".format(i)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        tmp_train_df = train_df.iloc[train_index]


        tmp_dev_df = train_df.iloc[dev_index]

        test_df.to_csv(DATA_DIR + "test.csv", header=None, index=0, sep='\t')
        if is_pse_label:
            pse_dir = "data_pse_{}/".format(i)
            pse_df = pd.read_csv(pse_dir + 'train.csv')

            tmp_train_df = pd.concat([tmp_train_df, pse_df], sort=False, ignore_index=True)

        tmp_train_df.to_csv(DATA_DIR + "train.csv", header=None, index=0, sep='\t', encoding='utf-8')
        tmp_dev_df.to_csv(DATA_DIR + "dev.csv", header=None, index=0, sep='\t', encoding='utf-8')
        print(tmp_train_df.shape, tmp_dev_df.shape)
        i += 1
generate_data(is_pse_label=False)