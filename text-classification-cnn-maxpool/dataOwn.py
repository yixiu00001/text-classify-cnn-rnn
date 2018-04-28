import tensorflow as tf
import numpy as np
import time
import os
import pickle
from gensim.models import Word2Vec

#根据word2ve模型构建word2indx和index2word词典
def create_voabulary(word2vec_model_path, name_scope=''):
    cache_dir =  './cache_vocabulary_label_pik/'
    cache_path = cache_dir + name_scope + "_word_voabulary.pik"
    #print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))
    # load the cache file if exists
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        model = Word2Vec.load(word2vec_model_path)
        print("vocabulary:", len(model.wv.vocab))
        for i, vocab in enumerate(model.wv.vocab):
            vocabulary_word2index[vocab] = i + 1
            vocabulary_index2word[i + 1] = vocab

        # save to file system if vocabulary of words is not exists.
        print(len(vocabulary_word2index))
        if not os.path.exists(cache_path):
            with open(cache_path, 'wb') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    lengths = [len(s) for s in sequences]
    #compyter the maxlen of sentense 
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    #print("maxlen=",maxlen)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        #如果是pre，则截取后面maxlen的词，如果是post，截取前maxlen的词
        if truncating == 'post':
            trunc = s[:maxlen]
        elif truncating == 'pre':
            trunc = s[-maxlen:]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def loadTrainOrTest_data(data_path, label_path, vocabulary_word2index, dictPath):
    #x_text = tf.Gfile(data_path)
    # Generate labels
    all_label = dict()
    if os.path.exists(dictPath):
        with open(dictPath, 'rb') as data_f:
            all_label = pickle.load(data_f)
    one_hot = np.identity(len(all_label))
    #print(one_hot)
    
    x_text = list(open(data_path,"r").readlines())
    x_temp = list([a.strip().split(" ") for a in x_text])
    x = [[a.strip() for a in b]  for b in x_temp]
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = vocabulary_word2index.get(x[i][j],0)
    x_train = np.array(x).tolist()
    y_text = list(open(label_path,"r").readlines())
    #获取lable种类数n，将label转化n位的编码，20180423
    y = [one_hot[ all_label[label.strip()] ] for label in y_text]
    y_train = np.array(y)
    #y_train = np.array([np.int32(a) for a in y_text])
    return x_train, y_train

def loadTrainOrTest_data_oneLabel(data_path, label_path, vocabulary_word2index):
    x_text = list(open(data_path,"r").readlines())
    x_temp = list([a.strip().split(" ") for a in x_text])
    x = [[a.strip() for a in b]  for b in x_temp]
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = vocabulary_word2index.get(x[i][j],0)
    x_train = np.array(x).tolist()
    y_text = list(open(label_path,"r").readlines())
    #获取lable种类数n，将label转化n位的编码，20180423
    y_train = np.array([np.int32(a) for a in y_text])
    return x_train, y_train

def loadTrainOrTest_data_oneLabel_Source(data_path, label_path, vocabulary_word2index):
    x_text = list(open(data_path,"r").readlines())
    #x_temp = list([a.strip().split(" ") for a in x_text])
    #x = [[a.strip() for a in b]  for b in x_temp]

    #x_train = np.array(x).tolist()
    y_text = list(open(label_path,"r").readlines())
    #获取lable种类数n，将label转化n位的编码，20180423
    y_train = np.array([np.int32(a) for a in y_text])
    return x_text, y_train


def assign_pretrained_word_embedding(sess, cnnDisease, word2vec_model,embed_size):
    #print("using pre-trained word emebedding.started.word2vec_model_path:", word2vec_model_path)
    word2vec_dict = {}
    vocab_size = len(word2vec_model.wv.index2word)
    print("vocab_size=",vocab_size)
    
    word_embedding_2dlist = [[]] * (vocab_size+1)  # create an empty word_embedding list.
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    count_exist = 0
    count_not_exist = 0
    word_embedding_2dlist[0] = np.random.uniform(-bound, bound, embed_size);
    for i, word in enumerate(word2vec_model.wv.vocab):
    #for i in range(vocab_size):
        #word = word2vec_model.wv.index2word[i]
        embedding = None
        try:
            embedding = word2vec_model.wv[word]
        except:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i+1] = embedding
            count_exist += 1
        else:
            word_embedding_2dlist[i+1] = np.random.uniform(-bound, bound, embed_size);
            count_not_exist += 1
        
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(cnnDisease.embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
    
def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    print("data_len=", data_len, " num_batch=", num_batch)
    x = np.array(x)
    y = np.array(y)
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]