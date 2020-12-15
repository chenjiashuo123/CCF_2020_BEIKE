# CCF BDCI 2020 房产行业聊天问答匹配 A榜47/2985

## 赛题描述详见：https://www.datafountain.cn/competitions/474

## 文件说明

**data**: 存放训练数据和测试数据以及预处理代码

**model_bert.py**: 网络模型结构定义

**adv_train.py**: 对抗训练代码

**run_bert_pse_adv.py**: 运行**bert-wwm + 对抗训练 + 伪标签**模型

**run_roberta_cls_pse_reinit_adv.py**: 运行**roberta-large last2embedding_cls + reinit + 对抗训练 + 伪标签**模型



## 个人方案
我的baseline是将query和answer拼接后传入预训练好的**bert**进行特征提取，之后将提取的特征传入一个**全连接层**，最后接一个**softmax**进行分类。

其中尝试的预训练模型有**bert**(谷歌)，**bert_wwm**(哈工大版本)，**roberta_large**(哈工大版本)，**xlnet**，**ernie**等，其中效果较好的有bert-wwm和roberta-large。之后在baseline的基础上进行了各种尝试，主要尝试有以下：<br>

| 模型                                                         | 线上F1 |
| ------------------------------------------------------------ | ------ |
| bert-wwm                                                     | 0.78   |
| bert-wwm + 对抗训练                                          | 0.783  |
| bert-wwm + 对抗训练 + 伪标签                                 | 0.7879 |
| roberta-large                                                | 0.774  |
| roberta-large + reinit + 对抗训练                            | 0.786  |
| roberta-large + reinit+对抗训练 + 伪标签                     | 0.7871 |
| roberta-large last2embedding_cls + reinit + 对抗训练 + 伪标签 | 0.7879 |

### 对抗训练

 其基本的原理呢，就是通过添加**扰动**构造一些对抗样本，放给模型去训练，**以攻为守**，提高模型在遇到对抗样本时的鲁棒性，同时一定程度也能提高模型的表现和泛化能力。 

参考链接：https://zhuanlan.zhihu.com/p/91269728

### 伪标签

将测试数据和预测结果进行拼接，之后当成训练数据传入到模型中重新进行训练。为了减少对训练数据的原始分布的影响并增加伪标签的置信度，我只在五个采用不同预训练模型的baseline预测一致的数据中采样了6000条测试数据加入到训练集进行训练。

### 重新初始化

参考链接：如何让Bert在finetune小数据集时更“稳”一点 https://zhuanlan.zhihu.com/p/148720604

大致思想是靠近底部的层（靠近input）学到的是比较通用的语义方面的信息，比如词性、词法等语言学知识，而靠近顶部的层会倾向于学习到接近下游任务的知识，对于预训练来说就是类似masked word prediction、next sentence prediction任务的相关知识。当使用bert预训练模型finetune其他下游任务（比如序列标注）时，如果下游任务与预训练任务差异较大，那么bert顶层的权重所拥有的知识反而会拖累整体的finetune进程，使得模型在finetune初期产生训练不稳定的问题。

因此，我们可以在finetune时，只保留接近底部的bert权重，对于靠近顶部的层的权重，可以重新随机初始化，从头开始学习。

在本次比赛中，我只对最后roberta-large的最后五层进行重新初始化。在实验中，我发现对于bert，重新初始化会降低效果，而roberta-large则有提升。

### bert 不同embedding和cls组合

思路主要是参考 CCF BDCI 2019 互联网新闻情感分析 复赛top1解决方案

参考链接：https://github.com/cxy229/BDCI2019-SENTIMENT-CLASSIFICATION

即对bert不同embedding进行组合后传入全连接层进行分类。该方案尝试时间较晚，只实验last2embedding_cls这种组合，结果也确实有提升。

### 模型融合

对于单模，我采用五折交叉验证，对每一个单模的五个模型结果，我尝试了相加融合和投票的方式，结果是融合相加的线上f1较高

对于不同模型，我也只是采用的相加融合的方式（由于时间问题没有尝试投票和stacking的方式）。最后a榜效果最好的是**bert-wwm + 对抗训练 + 伪标签、roberta-large + reinit+对抗训练 + 伪标签、roberta-large last2embedding_cls + reinit + 对抗训练 + 伪标签** 三个模型的融合，线上F1有 **0.7908** ， 排名47；B榜我尝试只对两个效果最好的模型进行融合，即 **bert-wwm + 对抗训练 + 伪标签**和 **last2embedding_cls + reinit + 对抗训练 + 伪标签**,最终F1为0.80，排名72。

### 总结

本次参加比赛完全是数据挖掘课程要求，也是我第一次参加大数据比赛。因为我的研究方向是图像，所以基本可以说是从零开始，写这个github只是想记录一下这一个月自己从零开始的参赛经历，也希望对同样参加类似比赛的新人有帮助。最后，希望看到了顺手给star，万分感谢。
