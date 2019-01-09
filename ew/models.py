import os

import keras
import numpy as np
from keras import Sequential
from keras.layers import Masking, Bidirectional, LSTM, BatchNormalization, TimeDistributed, Dense, Dropout, Flatten
from keras_preprocessing import sequence
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ew.defs import Example, Examples


class EWSpamClassifier:
    def __init__(self, model, exp_name, n_class=2):
        """

        :param model:
        :type model: Keras.model
        :param exp_name: 用来保存文件夹
        :type exp_name: str or None
        :param n_class: 有多少类 spam就是2类(loaded的模型可以为None)
        :type n_class: int or None
        """
        self.model = model
        self.exp_name = exp_name
        self.n_class = n_class
        self.enc = OneHotEncoder()
        self.enc.fit(np.array(list(range(self.n_class)), dtype=int).reshape(-1, 1))

    def to_one_hot(self, _labels):
        return self.enc.transform(_labels.reshape(-1, 1)).toarray()

    @classmethod
    def load_model(cls, model_path, n_class=2):
        model = keras.models.load_model(model_path)
        return cls(model, None, n_class)

    def infer(self, news_list):
        pass

    def train(self, examples_train, batch_size, epochs, max_len):
        """

        :param examples_train:
        :type examples_train: Examples
        :param batch_size:
        :type batch_size:
        :return:
        :rtype:
        """
        X, y = examples_train.to_keras_format(self.to_one_hot)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        X_train = sequence.pad_sequences(X_train, maxlen=max_len)
        X_valid = sequence.pad_sequences(X_valid, maxlen=max_len)

        model_dir = "./outputs/models/model_{en}".format(en=self.exp_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        log_dir = "./outputs/logs/log_{en}".format(en=self.exp_name)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=batch_size,
                                        write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None),
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(model_dir + "/weights.{epoch:02d}-{val_loss:.5f}.hdf5",
                                            monitor='val_loss', verbose=0, save_best_only=False,
                                            save_weights_only=False, mode='auto', period=1)
        ]

        self.model.save(model_dir + "/basic_model.h5")

        self.model.fit(X_train, y_train,
                       batch_size=batch_size, verbose=1,
                       epochs=epochs, callbacks=callbacks, validation_data=(X_valid, y_valid))


class EWSpamClassifierLSTM(EWSpamClassifier):
    """
    Bi-LSTM的文本分类器
    """

    def __init__(self, dim_input, exp_name, n_class=2):
        model = Sequential()
        # 似乎不需要masking 简单起见
        model.add(Masking(mask_value=0., input_shape=dim_input))
        model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
        model.add(Dense(500, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))
        model.add(Dense(n_class, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        print("Building model Done")
        super().__init__(model, exp_name, n_class)


if __name__ == '__main__':
    # BERT的dim 和 模型有关
    # 我现在的中文的是768 第一个参数是 这个序列(新闻的句子序列) 有多长 None的话就是不定长
    EWSpamClassifierLSTM((None, 768), "demo", 2)
