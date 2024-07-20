from keras.layers import Dense, LSTM
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Layer
import pandas as pd
from feature_select import Sample_generation, test_generation
import numpy as np
from sklearn.preprocessing import StandardScaler
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



def experiment(seed,path):
    # 读取数据


    test = np.array(test_generation(seed, path))
    probs = [0] * len(test)
    for i in range(1):
        data = np.array(Sample_generation(seed, path))

        LRI_name_z = pd.read_csv(path + 'LRI_name_z.csv', header=None, index_col=None).to_numpy()
        print("test shape", test.shape)

        Y = data[:, -1]
        X = data[:, :-1]

        std = StandardScaler()
        X = std.fit_transform(X)
        test = std.fit_transform(test)
        # # 归一化
        # mm = MinMaxScaler()
        # X = mm.fit_transform(X)
        # test = mm.fit_transform(test)

        X_train_lstm = X.reshape(X.shape[0], 1, X.shape[1])
        X_test_lstm = test.reshape(test.shape[0], 1, test.shape[1])

        input_shape = (1, 400)
        attention_units = 16
        # 定义固定学习率

        # 创建优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, epsilon=1e-8)
        # 创建模型
        model = create_lstm_model(input_shape, attention_units)

        # 编译模型
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_lstm, Y, epochs=400, batch_size=64, verbose=0)

        prob = model.predict(X_test_lstm)
        prob = np.array(list(chain.from_iterable(prob)))

        pr = pd.DataFrame(prob)
        pr.to_csv(path + "proba.csv", header=None, index=None)


        # probss = np.array(probs)
        # pr = pd.DataFrame(probs)
        # pr.to_csv(path + "proba.csv", header=None, index=None)

        probss = np.array(prob)
        pred = []
        for k in probss:
            if k == 1:
                pred.append(1)
            else:
                pred.append(0)

        pred = np.array(pred)
        pred_name = []
        for i in range(pred.shape[0]):
            if pred[i] == 1:
                pred_name.append(LRI_name_z[i])

        pred_name = pd.DataFrame(pred_name)
        pred_name.to_csv(path + 'proba_name.csv', header=None, index=None)

        pred = pd.DataFrame(pred)
        pred.to_csv(path + 'pred.csv', header=None, index=None)








if __name__ == "__main__":
    experiment(2, 'dataset/human/')
