#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TCNNConfig(object):
    """CNN配置参数"""

    # 模型参数 
    num_filters = 32          #filter数量
    num_classes = 14          # 类别数
    learning_rate = 0.01       #学习率
    batch_size = 64           #训练或者测试批大小
    sequence_length = 50       # 序列长度
    embed_size = 100          # 词向量维度
    num_epochs = 32           #迭代轮数
    decay_steps = 1500         #多少轮衰减学习率
    decay_rate = 0.9           #初始衰减值
    vocab_size = 5000       # 词汇表大小
    
    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False
    clip_gradiencets=0.5
    
    checkpoint_every = 4        #多少轮保存一次checkpoint  
    num_checkpoints = 16        #总共产出的checkpoint数量
    use_embedding = True        #是否使用预训练的embedding向量
    dropout_keep_prob = 0.5        # dropout保留比例
    validate_every = 4
    
    print_per_batch = 1000
    save_per_batch = 1000
    ckpt_dir = "./runs/cnn_disease_checkpointadmin/"
    filter_sizes = [3,4,5] #filter大小
    
    train_data_path = "../cnnDatasets/trainAdmin.feature"
    train_label_path = "../cnnDatasets/trainAdmin.label"
    test_data_path = "../cnnDatasets/testAdmin.feature"
    test_label_path = "../cnnDatasets/testAdmin.label"
    
    word2vec_model_path = "../modelKey/word2VecModelsn.bin15_100_1e-05_15"
    
class CNNDisease(object):
    def __init__(self, config):
        self.config = config
        self.initializer=tf.random_normal_initializer(stddev=0.1)
        
        #每行输入为词数为sequence_length（实验是21），每次输入行数不定，输出label行数也不定
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")
        #drop选取节点失活的概率
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.num_filters_total = self.config.num_filters*len(self.config.filter_sizes)
        
        self.cnn()
    def cnn(self):
        #cnn model
                # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable(
                "Embedding", 
                shape=[(self.config.vocab_size), self.config.embed_size], 
                initializer=self.initializer) 
            self.w_projection = tf.get_variable(
                "w_projection",
                shape=[self.num_filters_total, self.config.num_classes],
                initializer = self.initializer)
            self.b_projection = tf.get_variable(
                "b_projection",
                shape=[self.config.num_classes])
        with tf.name_scope("initialize"):
            self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False, name="learning_rate")
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
            self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
            #正态分布的张量,tf.random_normal_initializer((mean=0.0, stddev=1.0, seed=None, dtype=tf.float32),stddev 标准差
        with tf.name_scope("inference"):
            #1.get embedding word vector（word2vec trained）
            self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.sentence_embedded_expanded = tf.expand_dims(self.embedded_words, -1)
            #2.loop each filter size conv->relu->pool
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("convolution-pool-%s" %(filter_size)):
                    filter = tf.get_variable(
                        "filter-%s" %(filter_size), 
                        shape=[filter_size, self.config.embed_size,1, self.config.num_filters ],
                        initializer = self.initializer)
                    conv = tf.nn.conv2d(
                        self.sentence_embedded_expanded,
                        filter,
                        strides=[1,1,1,1],
                        padding = "VALID",
                        name = "conv")
                    b = tf.get_variable("b-%s" %(filter_size), shape = [self.config.num_filters])

                    h = tf.nn.relu(
                        tf.nn.bias_add(conv, b), 
                        "relu")
                    #pooled = tf.nn.avg_pool(
                    pooled = tf.nn.max_pool(
                        h, 
                        ksize=[1, self.config.sequence_length-filter_size+1,1,1],
                        strides=[1,1,1,1],
                        padding = "VALID",
                        name = "pool")
                    pooled_outputs.append(pooled)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.pooled_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.pooled_flat, keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            self.logits = tf.matmul(self.h_drop, self.w_projection)+self.b_projection
            self.y_pred_cls = tf.argmax(self.logits, 1, name = "predictions")
            
        with tf.name_scope("loss"):
            l2_lambda=0.0001
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_y,
                logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name] )* l2_lambda
            
            self.loss = loss + l2_loss
        with tf.name_scope("accuracy"):
            prediction_now = tf.equal(tf.cast(self.y_pred_cls,tf.int32), self.input_y)
            self.acc = tf.reduce_mean(tf.cast(prediction_now, tf.float32), name="accuracy")
        
        with tf.name_scope("train"):
            #根据设定调节learning_rate
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                self.config.decay_steps, self.config.decay_rate, staircase=True)
            self.train_op = tf.contrib.layers.optimize_loss(self.loss, 
                global_step = self.global_step,
                learning_rate = self.learning_rate,
                optimizer="Adam",
                clip_gradients=self.config.clip_gradiencets)
            