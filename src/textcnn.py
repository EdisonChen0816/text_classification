# encoding=utf-8
import tensorflow as tf
from pyhanlp import HanLP
import numpy as np
import random


'''
word2vec + textcnn
'''


class TextCNN:

    def __init__(self, logger, train_data, eval_data, max_len, w2v, filters, kernel_size, pool_size, strides, loss, rate, epoch,
                 batch_size, dropout, model_path, summary_path, tf_config, dim=300):
        self.logger = logger
        self.train_data = train_data
        self.eval_data = eval_data
        self.max_len = max_len
        self.w2v = w2v
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.loss = loss
        self.rate = rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.dropout = dropout
        self.model_path = model_path
        self.summary_path = summary_path
        self.tf_config = tf_config
        self.dim = dim
        self.label = {}

    def get_input_feature(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                embedding = []
                text, tag = line.replace('\n', '').split('\t')
                if tag not in self.label:
                    self.label[tag] = len(self.label)
                for term in HanLP.segment(text):
                    word = term.word
                    if word in self.w2v:
                        embedding.append(self.w2v[word])
                if len(embedding) < self.max_len:
                    for i in range(self.max_len - len(embedding)):
                        embedding.append([0] * self.dim)
                else:
                    embedding = embedding[: self.max_len]
                data.append([embedding, self.label[tag]])
        return np.asarray(data)

    def batch_yield(self, data, shuffle=True):
        if shuffle:
            random.shuffle(data)
        seqs, labels = [], []
        for sent, tag in data:
            if len(seqs) == self.batch_size:
                yield np.asarray(seqs), np.asarray(labels)
                seqs, labels = [], []
            seqs.append(sent)
            labels.append(tag)
        if len(seqs) != 0:
            yield np.asarray(seqs), np.asarray(labels)

    def conv_net(self, x, dropout):
        conv1 = tf.layers.conv2d(x, self.filters, kernel_size=(self.kernel_size, self.dim), strides=self.strides, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(self.pool_size, 1), strides=self.strides)
        fc1 = tf.layers.flatten(pool1, name="fc1")
        fc2 = tf.layers.dense(fc1, 128)
        fc3 = tf.layers.dropout(fc2, rate=dropout)
        out = tf.layers.dense(fc3, len(self.label))
        return out

    def fit(self):
        data = self.get_input_feature(self.train_data)
        x = tf.placeholder(shape=[None, self.max_len, self.dim], dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None], dtype=tf.float32, name='y')
        x_input = tf.reshape(x, shape=[-1, self.max_len, self.dim, 1])
        dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        logits = self.conv_net(x_input, drop_rate)
        tf.add_to_collection("logits", logits)
        cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(cross_loss)
        if 'adam' == self.loss.lower():
            optim = tf.train.AdamOptimizer(self.rate).minimize(loss)
        elif 'sgd' == self.loss.lower():
            optim = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        else:
            optim = tf.train.GradientDescentOptimizer(self.rate).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                for step, (seqs, labels) in enumerate(self.batch_yield(data)):
                    _, curr_loss = sess.run([optim, loss], feed_dict={x: seqs, y: labels, dropout: self.dropout})
                    if step % 10 == 0:
                        print("epoch:%d, batch: %d, current loss: %f" % (i, step+1, curr_loss))
            saver.save(sess, self.model_path)
            tf.summary.FileWriter(self.summary_path, sess.graph)
            self.evaluate(sess, x, y, dropout, logits)

    def evaluate(self, sess, x, y, dropout, logits):
        eval_data = self.get_input_feature(self.eval_data)
        tp = 0  # 正类判定为正类
        fp = 0  # 负类判定为正类
        fn = 0  # 正类判定为负类
        for _, (seqs, labels) in enumerate(self.batch_yield(eval_data)):
            preds = sess.run(logits, feed_dict={x: seqs, y: labels, dropout: 1.0})
            tp += len(labels & preds)
            fp += len(preds - labels)
            fn += len(labels - preds)
        recall = tp / (tp + fn + 0.1)
        precision = tp / (tp + fp + 0.1)
        f1 = (2 * recall * precision) / (recall + precision + 0.1)
        self.logger.info('eval recall:' + str(recall) + ', eval precision:' + str(precision) + ', eval f1:' + str(f1))

    def load(self, path):
        self.pred_sess = tf.Session(config=self.tf_config)
        saver = tf.train.import_meta_graph(path + '/model.meta')
        saver.restore(self.pred_sess, tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('x:0')
        self.y = graph.get_tensor_by_name('y:0')
        self.dropout = graph.get_tensor_by_name('dropout:0')
        self.logits = tf.get_collection('logits')

    def close(self):
        self.pred_sess.close()

    def _predict_text_process(self, text):
        embedding = []
        for term in HanLP.segment(text):
            word = term.word
            if word in self.w2v:
                embedding.append(self.w2v[word])
        if len(embedding) < self.max_len:
            for i in range(self.max_len - len(embedding)):
                embedding.append([0] * self.dim)
        else:
            embedding = embedding[: self.max_len]
        return np.asarray([embedding])

    def predict(self, text):
        seq = self._predict_text_process(text)
        pred, _ = self.pred_sess.run(self.logits, feed_dict={self.x: seq, self.dropout: 0})
        return pred