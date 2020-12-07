import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def data_format(queryfile, replyfile, savefile, mode='train'):
    if mode == 'train':
        encode = 'utf-8'
    else:
        encode = 'GBK'
    query_data = pd.read_csv(queryfile, encoding=encode, sep='\t', header=None, names=['query_id', 'query'])
    query_data.set_index('query_id', inplace=True)
    print(len(query_data))
    print(query_data[:5])
    if mode == 'train':
        columns = ['query_id', 'reply_id', 'reply', 'label']
    else:
        columns = ['query_id', 'reply_id', 'reply']
    reply_data = pd.read_csv(replyfile, encoding=encode, sep='\t', header=None, names=columns)
    print(len(reply_data))
    print(reply_data[:5])
    print(reply_data['query_id'].apply(lambda x: query_data.loc[x, 'query']))
    reply_data['query'] = reply_data['query_id'].apply(lambda x: query_data.loc[x, 'query'])
    print(reply_data[:5])
    if mode == 'train':
        save_columns = ['query_id', 'reply_id', 'query', 'reply', 'label']
    else:
        save_columns = ['query_id', 'reply_id', 'query', 'reply']
    reply_data.loc[:, save_columns].to_csv(savefile, sep='\t', index=False, header=False)


if __name__ == '__main__':
    data_format(queryfile='train/train.query.tsv', replyfile='train/train.reply.tsv', savefile='process_data/train.tsv')
    data_format(queryfile='test/test.query.tsv', replyfile='test/test.reply.tsv', savefile='process_data/test.tsv', mode='test')