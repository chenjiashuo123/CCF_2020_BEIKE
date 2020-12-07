# CCF BDCI 2020 房产行业聊天问答匹配个人方案

## 赛题描述详见：https://www.datafountain.cn/competitions/474

## 个人方案
我的baseline是将query和answer拼接后传入预训练好的bert进行特征提取，之后将提取的特征传入一个全连接层进行分类。其中尝试的预训练模型有bert(谷歌)，bert_wwm(哈工大版本)，roberta_large(哈工大版本)，xlnet，ernie等，其中效果较好的有bert-wwm和roberta-large。之后在baseline的基础上进行了各种尝试，主要尝试有以下：<br>

| 模型                                                         | 线上F1 |
| ------------------------------------------------------------ | ------ |
| bert-wwm                                                     | 0.78   |
| bert-wwm + 对抗训练                                          | 0.783  |
| bert-wwm + 对抗训练 + 伪标签                                 | 0.7879 |
| roberta-large                                                | 0.774  |
| roberta-large + reinit + 对抗训练                            | 0.786  |
| roberta-large + reinit+对抗训练 + 伪标签                     | 0.7871 |
| roberta-large last2embedding_cls + reinit + 对抗训练 + 伪标签 | 0.7879 |


