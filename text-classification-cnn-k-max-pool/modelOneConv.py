import tensorflow as tf
printDebug=0
class TCNN_K_Config(object):
    """RNN配置参数"""

    # 模型参数
    batch_size = 64  #训练或者测试批大小
    sentence_length = 50       # 序列长度
    num_filters = [6, 14]         #filter数量
    use_embedding = True
    dropout_keep_prob = 0.5
    num_classes = 14          # 类别数
    learning_rate = 0.01       #学习率
    top_k = 4
    k1 = 26
    vocab_size = 5000       # 词汇表大小    
    embed_size = 100          # 词向量维度
    num_epochs = 32           #迭代轮数
    decay_steps = 1500         #多少轮衰减学习率
    decay_rate = 0.9           #初始衰减值
    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False
    clip_gradiencets=0.5
    print_per_batch = 1000
    save_per_batch = 1000

    ws = [7,5]
    num_hidden = 100
    
    train_data_path = "../cnnDatasets/trainAdmin.feature"
    train_label_path = "../cnnDatasets/trainAdmin.label"
    test_data_path = "../cnnDatasets/testAdmin.feature"
    test_label_path = "../cnnDatasets/testAdmin.label"
    
    word2vec_model_path = "../modelKey/word2VecModelsn.bin15_100_1e-05_15"

class CNN_K_MAXPOOL_DISEASE(object):

    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sentence_length],name="sent")
        self.input_y = tf.placeholder(tf.int32, [None], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.cnn()
    def cnn(self):
        with tf.name_scope("embedding_layer"):
            #self.W = tf.Variable(tf.random_uniform([self.vocab_size+1, self.embed_dim], -1.0, 1.0), name="embed_W")
            self.embedding = tf.get_variable("embed_W", shape=[self.config.vocab_size, self.config.embed_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
          
            #[batch_size, sentence_length, embed_dim, 1]
        with tf.name_scope("initialize"):
            self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False, name="learning_rate")
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
            self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
            #正态分布的张量,tf.random_normal_initializer((mean=0.0, stddev=1.0, seed=None, dtype=tf.float32),stddev 标准差
        def init_weights(shape, name):
            print("weight shape:",shape)
            #tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
            #return tf.get_variable(name, shape = shape, initializer =tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32) )
            return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32), name=name)
        with tf.name_scope("param_initialize"):
            self.W1 = init_weights([self.config.ws[0], self.config.embed_size, 1, self.config.num_filters[0]], "W1")
            #b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters[0], self.embed_dim]), "b1")#和常规的cnn的b相比，多了embed_dim一个维度
            self.b1 = tf.get_variable("b1", shape= [self.config.num_filters[0], self.config.embed_size])

            #W2 = init_weights([self.ws[1], int(self.embed_dim/2), self.num_filters[0], self.num_filters[1]], "W2")
            #b2 = tf.Variable(tf.constant(0.1, shape=[self.num_filters[1], self.embed_dim]), "b2")
            #b2 = tf.get_variable("b2", shape=[self.num_filters[1], self.embed_dim])

            self.Wh = init_weights([int(self.config.top_k*self.config.embed_size*self.config.num_filters[0]/2), self.config.num_hidden], "Wh")
            #bh = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), "bh")
            self.bh = tf.get_variable("bh", shape=[self.config.num_hidden])

            self.Wo = init_weights([self.config.num_hidden, self.config.num_classes], "Wo")
            
        with tf.name_scope("inference"):
            print("input shape0=",self.input_x.get_shape())
            self.sent_embed = tf.nn.embedding_lookup(self.embedding, self.input_x)#[batch_size, sentence_length, embed_dim]
            if printDebug ==0:
                print("sent_embed shape=", self.sent_embed)
            #input_x = tf.reshape(sent_embed, [batch_size, -1, embed_dim, 1])
            self.sentence_embedded_expanded = tf.expand_dims(self.sent_embed, -1)
            conv1 = self.per_dim_conv_layer(self.sentence_embedded_expanded, self.W1, self.b1)
            if printDebug ==0:
                print("conv1-con shape=", conv1.get_shape())
            fold = self.fold_k_max_pooling(conv1, self.config.top_k)
            if printDebug==0:
                print("conv1-kemax-pool shape=", conv1.get_shape())
            #conv2 = self.per_dim_conv_layer(conv1, W2, b2)
            #if printDebug==0:
            #    print("conv2 shape=",conv2.get_shape())
            #fold = self.fold_k_max_pooling(conv2, self.top_k)
            #if printDebug==0:
            #    print("fold shape=", fold.get_shape())
            #100-embed_size, 6-num_filters, 2- for i in range(0, len(input_unstack), 2):
            fold_flatten = tf.reshape(fold, [-1, int(self.config.top_k*100*6/2)])

            if printDebug==0:
                print( "trained shape=",fold_flatten.get_shape())
            self.out = self.full_connect_layer(fold_flatten, self.Wh, self.bh, self.Wo, self.dropout_keep_prob)
            if printDebug==0:
                print( "out shape=",self.out.get_shape())
        
        with tf.name_scope("output"):
            self.y_pred_cls = tf.argmax(self.out, 1, name = "predictions")     
        with tf.name_scope("loss"):
            l2_lambda=0.0001
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_y,
                logits=self.out)
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
        

    def per_dim_conv_k_max_pooling_layer(self, x, w, b, k):
        self.k1 = k
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv_k_max_pooling"):
            for i in range(self.embed_size):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])
                #conv:[batch_size, sent_length+ws-1, num_filters]
                conv = tf.reshape(conv, [self.batch_size, self.num_filters[0], self.sentence_length])#[batch_size, sentence_length, num_filters]
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.reshape(values, [self.batch_size, k, self.num_filters[0]])
                #k_max pooling in axis=1
                convs.append(values)
            conv = tf.stack(convs, axis=2)
        #[batch_size, k1, embed_size, num_filters[0]]
        #print conv.get_shape()
        return conv

    def per_dim_conv_layer(self, x, w, b):
        if printDebug==0:
            print("input shape:", x.get_shape())
        #x : [None, config.sentence_length, self.embed_dim]
        input_unstack = tf.unstack(x, axis=2)
        if printDebug==0:
            print("input_unstack shape:",len(input_unstack))
        #w : [self.ws[0], self.embed_dim, 1, self.num_filters[0]]
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv"):
            for i in range(len(input_unstack)):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return conv

    def fold_k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        return fold

    def full_connect_layer(self, x, w, b, wo, dropout_keep_prob):
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(x, w) + b)
            h = tf.nn.dropout(h, dropout_keep_prob)
            o = tf.matmul(h, wo)
        return o
   
    