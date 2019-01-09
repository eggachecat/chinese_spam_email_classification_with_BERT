from random import shuffle

import numpy as np


class RawExample:
    def __init__(self, news, label):
        """

        :param news: 新闻
        :type news: str
        :param label:
        :type label: int
        """
        self.news = news  # type: str
        self.label = label  # type:int


class RawExamples:
    def __init__(self, raw_examples=None):
        """

        :param raw_examples:
        :type raw_examples: list of RawExample
        """
        if raw_examples is None:
            raw_examples = []
        self.raw_examples = raw_examples

    @classmethod
    def load_from_folder(cls, file_path):
        pass


class Example:
    def __init__(self, news_encoding_sequence, label):
        """

        :param news_encoding_sequence: [SE_0, SE_1, SE_2, ...]
        :type news_encoding_sequence: np.ndarray
        :param label:
        :type label: int
        """
        self.news_encoding_sequence = news_encoding_sequence  # type: list
        self.label = label  # type:int


class Examples:
    def __init__(self, examples=None):
        """

        :param examples:
        :type examples: list of Example
        """
        if examples is None:
            examples = []
        self.examples = examples

    def add(self, example):
        self.examples.append(example)

    def __iter__(self):
        for example in self.examples:
            yield example

    def __len__(self):
        return len(self.examples)

    def shuffle(self):
        shuffle(self.examples)

    def to_keras_format(self, one_hot_encoder):
        X_train = []
        y_train = []
        for example in self.examples:
            X_train.append(example.news_encoding_sequence)
            # print(one_hot_encoder(np.array([[example.label]]))[0])
            y_train.append(one_hot_encoder(np.array([[example.label]]))[0])
        return np.array(X_train), np.array(y_train)
