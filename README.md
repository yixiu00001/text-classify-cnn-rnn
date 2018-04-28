# text-classify-cnn-rnn
基于cnn（maxPool/k-maxPool）和rnn的文本分类
本文主要介绍在文本分类中，使用CNN网络和RNN网络的实践，其中CNN又分为maxPool和k-maxpool。可以直接在juputer执行。
## 1.CNN+maxPool
text-classification-cnn-maxpool
该工程为cnn-maxpool相关代码。

```
 dataOwn.py
```
主要包括基于word2vec的embedding方法构建词和index的映射词典，词长不到设定值的打padding以及数据的载入和batch_iter

```
CNNDiseaseModel.py 
```
主要是CNN网络相关的变量初始化及网络构造。
重点看placeholder/inference等部分，在计算损失部分，由于目前的label只有一个数值，非onehot类型，所以调用sparse_softmax_cross_entropy_with_logits接口。
![image](D:\source\深度学习经典书籍\图片\ner\loss.bmp)

```
CNNDiseaseModelTrain.ipynb
```
这个文件是训练模型的文件，在train部分是整个训练的逻辑。


```
CNNDiseaseModelPredict.ipynb
```
这个文件是对已经训练好的模型，进行结果测试，提供了输入一段文本进行测试的接口和输入一个测试文件地址进行测试的接口。
## 2.CNN-k-max-pool
这里和上面工程的区别是使用了k-max pool，但是本实验中效果和max-pool差不多。
同样包含几个文件

```
dataOwn.py
modelOneConv.py
trainWord2vec.ipynb
predictWord2vec.ipynb
```

# 3.text-classification-rnn
这个文件是基于rnn实现的分类，可以选择使用LSTM或者GRU

```
rnn_model_oneLable.py
train_rnn_oneLable.ipynb
predict_rnn_oneLablelNew.ipynb
```
这三个文件对应的数据的label是一个数字，如0 ，1 ，2这种类型

```
rnn_model_onehotLable.py
train_rnn_onehotLable.ipynb
