数据集来自百度webQA：http://idl.baidu.com/WebQA.html
现请移步dureader数据集：http://ai.baidu.com/broad/introduction?dataset=dureader
论文参考：Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering
相似论文：Reading Wikipedia to Answer Open-Domain Questions

QA模型：
对question使用q_lstm建模，并加入attention机制，将问题向量和文档词向量拼接起来，一起输入文档建模模型。对evidence使用e_lstm建模，softmax进行序列标注。
（crf序列标注的代码有待修改）
变长lstm的实现这里写得太笨了，应该把mask改为length，只传length即可。

文档检索：爬虫百度知道。




