
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Layer
import pandas as pd
from feature_select import Sample_generation800
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from itertools import chain

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SelfAttention(Layer):
    def __init__(self, attention_units):
        super(SelfAttention, self).__init__()
        self.attention_units = attention_units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.attention_units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.attention_units,),
                                 initializer='zeros',
                                 trainable=True)
        self.V = self.add_weight(shape=(self.attention_units, 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        query = tf.matmul(inputs, self.W) + self.b
        score = tf.matmul(tf.nn.tanh(query), self.V)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# 创建一个包含自注意力机制的LSTM模型
def create_lstm_model(input_shape, attention_units):
    model = tf.keras.Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=input_shape))
    model.add(SelfAttention(attention_units))
    model.add(Dense(256, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))

    model.add(Dense(1, activation='sigmoid'))
    return model

def main(seed,path):

    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    itime = time.time()
    for i in range(20):

        print("*************%d**************"%(i+1))
        data = np.array(Sample_generation800(seed, path))

        X = data[:, :-1]
        Y = data[:, -1]

        std = StandardScaler()
        X = std.fit_transform(X)


        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


            input_shape = (1, X.shape[1])
            attention_units = 16
            # 定义固定学习率


            # 创建优化器
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, epsilon=1e-8)
            # 创建模型
            model = create_lstm_model(input_shape, attention_units)

            # 编译模型
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, verbose=0)

            prob = model.predict(X_test_lstm)
            prob_lstm = np.array(list(chain.from_iterable(prob)))
            # pred = np.argmax(prob, axis=1)

            pred = []
            for k in prob_lstm:
                if k > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
            pred = np.array(pred)



            sum_acc += accuracy_score(y_test, pred)
            sum_pre += precision_score(y_test, pred)
            sum_recall += recall_score(y_test, pred)
            sum_f1 += f1_score(y_test, pred)

            fpr, tpr, thresholds = roc_curve(y_test, prob)
            prec, rec, thr = precision_recall_curve(y_test, prob)
            sum_AUC += auc(fpr, tpr)
            sum_AUPR += auc(rec, prec)

        m_acc.append(sum_acc / 5)
        m_pre.append(sum_pre / 5)
        m_recall.append(sum_recall / 5)
        m_F1.append(sum_f1 / 5)
        m_AUC.append(sum_AUC / 5)
        m_AUPR.append(sum_AUPR / 5)

        print("****** The %d 5-fold validation performance is：" % (i + 1))
        print("precision:%.4f+%.4f" % (sum_pre / 5, np.std(np.array(m_pre))))
        print("recall:%.4f+%.4f" % (sum_recall / 5, np.std(np.array(m_recall))))
        print("accuracy:%.4f+%.4f" % (sum_acc / 5, np.std(np.array(m_acc))))
        print("F1 score:%.4f+%.4f" % (sum_f1 / 5, np.std(np.array(m_F1))))
        print("AUC:%.4f+%.4f" % (sum_AUC / 5, np.std(np.array(m_AUC))))
        print("AUPR:%.4f+%.4f" % (sum_AUPR / 5, np.std(np.array(m_AUPR))))
        print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime) / 60))
        print("********************The %d 5-fold validation is over" % (i + 1))

    print("precision:%.4f+%.4f" % (np.mean(m_pre), np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f" % (np.mean(m_recall), np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f" % (np.mean(m_acc), np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f" % (np.mean(m_F1), np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f" % (np.mean(m_AUC), np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f" % (np.mean(m_AUPR), np.std(np.array(m_AUPR))))
    print(' Total code computed. Time: {}m'.format((time.time() - itime) / 60))
    print("******End of code ******")


if __name__ == "__main__":
    main(1, 'dataset/human/')

