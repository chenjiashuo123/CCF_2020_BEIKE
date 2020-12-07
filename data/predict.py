import pandas as pd
import numpy as np
import os
from pprint import pprint
from sklearn.metrics import f1_score
import csv
import random
from sklearn.model_selection import KFold, GroupKFold

# DATA_DIR = 'data_KFold'
# dev_label = []
# # sub = pd.read_csv(DATA_DIR+'\data_origin_0\dev.csv', sep='\t', names=['query_id', 'reply_id', 'query', 'reply','label'])
# filename = DATA_DIR+'\data_origin_0\dev.csv'
# with open(filename, encoding='utf-8') as f:
#     f_csv = csv.reader(f, delimiter='\t')
#     for line in f_csv:
#         dev_label.append(line)
# name = ['query_id', 'reply_id', 'query', 'reply','label']
# test=pd.DataFrame(columns=name,data=dev_label)
# test.to_csv('test.csv', encoding='utf-8', index=False, sep="\t", header=None)
# print(dev_label)



def cat_dev(DATA_DIR):
    files = os.listdir(DATA_DIR)
    dev_label = []
    for i in files:
        filename = DATA_DIR+i+'/dev.csv'
        with open(filename, encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter='\t')
            for line in f_csv:
                dev_label.append(line)
    name = ['query_id', 'reply_id', 'query', 'reply', 'label']
    test = pd.DataFrame(columns=name, data=dev_label)
    test.to_csv('dev_all.csv', encoding='utf-8', index=False, sep="\t", header=None)

'''拼接fold'''
def cat_flod(DATA_DIR):
    files = os.listdir(DATA_DIR)
    for i in range(5):
        dev_pred = []
        for file in files:
            datas = os.listdir(DATA_DIR+'/'+file+'/logits')
            filename = DATA_DIR+'/'+file+'/logits/'+datas[i]
            with open(filename, encoding='utf-8') as f:
                f_csv = csv.reader(f, delimiter='\t')
                for line in f_csv:
                    dev_pred.append(line)
        name = ['query_id', 'reply_id', 'pred', 'label', 'logits_0', 'logits_1']
        test = pd.DataFrame(columns=name, data=dev_pred)
        test[['logits_0', 'logits_1']].to_csv('dev_logits_{}.csv'.format(i), encoding='utf-8', index=False, sep="\t", header=None)
        test[['label', 'pred']].to_csv('dev_pred_{}.csv'.format(i), encoding='utf-8', index=False, sep="\t", header=None)


'''使用相加的方式融合dev'''
def fusion_dev():
    res_proba = np.zeros((21585, 2))
    for i in range(5):
        filename = 'dev_logits_{}.csv'.format(i)
        dev_pred = []
        with open(filename, encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter='\t')
            for line in f_csv:
                dev_pred.append(line)
            dev_pred = np.array(dev_pred, dtype=float)
            res_proba += dev_pred
    pred = np.argmax(res_proba, axis=1)
    filename = 'dev_pred_0.csv'
    dev_label = []
    with open(filename, encoding='utf-8') as f:
        f_csv = csv.reader(f, delimiter='\t')
        for line in f_csv:
            dev_label.append(line[0])
    dev_label = np.array(dev_label, dtype = float)
    print(f1_score(dev_label, pred))

'''使用相加的方式融合test'''
def fusion_test():
    res_proba = np.zeros((53757, 2))
    data_dir = 'fusion/'
    dir_files = os.listdir(data_dir)
    for files in dir_files:
        data_file = os.listdir(data_dir+files)
        for file in data_file:
            filename = data_dir + files + '/' + file
            dev_pred = []
            with open(filename, encoding='utf-8') as f:
                f_csv = csv.reader(f, delimiter='\t')
                for line in f_csv:
                    dev_pred.append(line[3:5])
                    # print(dev_pred)
                dev_pred = np.array(dev_pred, dtype=float)
                res_proba += dev_pred
    df = pd.read_csv('test_res/res/bert_adv_submission.csv', encoding="utf-8", sep='\t', header=None)
    df.columns = ['query_id', 'reply_id', 'label']
    df['label_fusion'] = np.argmax(res_proba, axis=1)
    df[['query_id', 'reply_id', 'label_fusion']].to_csv( "fusion_2_submission.csv", index=False, sep="\t",
                                                 header=None)



def fusion_sub_test():
    res_proba = np.zeros((53757, 2))
    data_dir = 'fusion/roberta_cls_pse_adv'
    dir_files = os.listdir(data_dir)
    for file in dir_files:
        filename = data_dir + '/' + file
        dev_pred = []
        with open(filename, encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter='\t')
            for line in f_csv:
                dev_pred.append(line[3:5])
                # print(dev_pred)
            dev_pred = np.array(dev_pred, dtype=float)
            res_proba += dev_pred
    df = pd.read_csv('test_res/res/bert_adv_submission.csv', encoding="utf-8", sep='\t', header=None)
    df.columns = ['query_id', 'reply_id', 'label']
    df['label_fusion'] = np.argmax(res_proba, axis=1)
    df[['query_id', 'reply_id', 'label_fusion']].to_csv( "roberta_cls_pse_adv.csv", index=False, sep="\t",
                                                 header=None)
def cat_test():
    test_data = pd.read_csv('data_KFold/data_origin_0/test.csv', encoding="utf-8", sep='\t', header=None)
    test_data.columns = ['query_id', 'reply_id', 'query', 'reply']
    df0 = pd.read_csv('test_res/res/bert_adv_submission.csv', encoding="utf-8", sep='\t', header=None)
    df0.columns = ['query_id', 'reply_id', 'label']
    df1 = pd.read_csv('test_res/res/bert_base_submission.csv', encoding="utf-8", sep='\t', header=None)
    df1.columns = ['query_id', 'reply_id', 'label']
    df2 = pd.read_csv('test_res/res/ernie_submission.csv', encoding="utf-8", sep='\t', header=None)
    df2.columns = ['query_id', 'reply_id', 'label']
    df3 = pd.read_csv('test_res/res/roberta_submission.csv', encoding="utf-8", sep='\t', header=None)
    df3.columns = ['query_id', 'reply_id', 'label']
    df4 = pd.read_csv('test_res/res/xlnet_submission.csv', encoding="utf-8", sep='\t', header=None)
    df4.columns = ['query_id', 'reply_id', 'label']
    test_data['label_0'] =  df0['label']
    test_data['label_1'] =  df1['label']
    test_data['label_2'] =  df2['label']
    test_data['label_3'] =  df3['label']
    test_data['label_4'] =  df4['label']
    test_data[['query_id', 'reply_id', 'query', 'reply', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4']].to_csv( "all_test_pred.csv", index=False, sep="\t",
                                                 header=None)

def aug_data():
    filename = "test_res/all_test_pred.csv"
    with open(filename, encoding='utf-8') as f:
        f_csv = csv.reader(f, delimiter='\t')
        test_all_true = []
        for line in f_csv:
            sum = int(line[4])+int(line[5])+int(line[6])+int(line[7])+int(line[8])
            if sum == 5 or sum == 0:
                test_all_true.append(line[0:5])

    random.seed(324)
    # test_true_sample = random.sample(test_all_true, 10000)
    test_true_sample = test_all_true[0:10000]
    ids = [x[0] for x in test_true_sample]
    # print(test_true_sample)
    kf = GroupKFold(n_splits=5)
    i = 0
    for train, test in kf.split(test_true_sample, groups=ids):
        DATA_DIR = "./data_KFold/data_origin_{}/".format(i)
        filename = DATA_DIR + 'train.csv'
        save_name = DATA_DIR + 'train_pse_4.csv'
        train_aug = []
        with open(filename, encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter='\t')
            for line in f_csv:
                train_aug.append(line)
            for item in test:
                train_aug.append(test_true_sample[item])
        name = ['query_id', 'reply_id', 'query', 'reply', 'label']
        train_aug = pd.DataFrame(columns=name, data=train_aug)
        train_aug.to_csv(save_name, encoding='utf-8', index=False, sep="\t",
                             header=None)
        i = i+1




    # name = ['query_id', 'reply_id', 'query', 'reply', 'label']
    # test_all_true = pd.DataFrame(columns=name, data=test_all_true)
    # test_all_true.to_csv('test_all_true.csv', encoding='utf-8', index=False, sep="\t",
    #                                       header=None)
    # test_all_false = pd.DataFrame(columns=name, data=test_all_false)
    # test_all_false.to_csv('test_all_false.csv', encoding='utf-8', index=False, sep="\t",
    #                                       header=None)


def vote(DATA_DIR):
    return

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs)






#
if __name__ == "__main__":
    # DATA_DIR = './data_KFold/'
    # cat_dev(DATA_DIR)
    # DATA_DIR = './result/'
    # cat_flod(DATA_DIR)

    # filename = 'dev_pred_3.csv'
    # dev_label = []
    # dev_pred = []
    # with open(filename, encoding='utf-8') as f:
    #     f_csv = csv.reader(f, delimiter='\t')
    #     for line in f_csv:
    #         dev_label.append(line[0])
    #         dev_pred.append(line[1])
    # dev_label = np.array(dev_label, dtype = float)
    # dev_pred = np.array(dev_pred, dtype = float)
    # print( f1_score(dev_label, dev_pred))
    fusion_test()
