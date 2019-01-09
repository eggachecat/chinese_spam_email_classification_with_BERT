"""
由于data source不同
数据格式肯定也不同

这边测试的是
    https://github.com/shijing888/BayesSpam
的数据集
    这个数据集是邮件
    邮件有个特点是一行一行的
    所有每一行当个sentence了这边
直接看可能打不开(编码问题)
    把文件拖到浏览器就能看了
"""
import os
import re

from ew.defs import Examples, Example
from ew.tools import BERTEncoder
import numpy as np


class shijing888FormatData:
    def __init__(self, ip=None, sentence_delimiters=None,
                 normal_file_folder=None, spam_file_folder=None, test_file_folder=None, path_to_load=None):
        self.normal_examples = None
        self.spam_examples = None
        self.test_examples = None
        if path_to_load is None:
            self.encoder = BERTEncoder(ip, sentence_delimiters)
            self.normal_file_folder = normal_file_folder
            self.spam_file_folder = spam_file_folder
            self.test_file_folder = test_file_folder
            # 1表示是spam
            self.normal_examples = self.get_raw_examples(self.normal_file_folder, lambda x: 0)
            self.spam_examples = self.get_raw_examples(self.spam_file_folder, lambda x: 1)
            self.test_examples = self.get_raw_examples(self.test_file_folder, lambda x: 0 if int(x) <= 1000 else 1)
        else:
            self.normal_examples = Examples()
            self.spam_examples = Examples()
            self.test_examples = Examples()
            self.load_from_path(path_to_load)

    def get_examples_train(self):
        examples_train = Examples()
        for example in self.spam_examples:
            examples_train.add(example)
        for example in self.normal_examples:
            examples_train.add(example)
        return examples_train

    def get_raw_examples(self, base_folder, label_func):
        """

        :param base_folder:
        :type base_folder: 文件文件夹
        :param label_func:
        :type label_func: 通过file_name判断
        :return:
        :rtype: Examples
        """
        examples = Examples()
        file_name_list = os.listdir(base_folder)
        # 测试集的label是这样的: name < 1000 是正常邮件
        for i, file_name in enumerate(file_name_list):
            if i % 100 == 0 and i > 0:
                print("{} done".format(i))

            sentences = []
            for line in open(os.path.join(base_folder, file_name)):
                # 过滤掉非中文字符
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                if line != "":
                    sentences.append(line)
            if len(sentences) > 0:
                encoding = self.encoder.encode_sentences(sentences)
                label = label_func(file_name)
                examples.add(Example(encoding, label))
        return examples

    def save_to_path(self, path_to_save):
        np.save(os.path.join(path_to_save, "normal.npy"),
                np.array([[example.news_encoding_sequence, example.label] for example in self.normal_examples]))
        np.save(os.path.join(path_to_save, "spam.npy"),
                np.array([[example.news_encoding_sequence, example.label] for example in self.spam_examples]))
        np.save(os.path.join(path_to_save, "test.npy"),
                np.array([[example.news_encoding_sequence, example.label] for example in self.test_examples]))

    def load_from_path(self, path_to_load):
        for example in np.load(os.path.join(path_to_load, "normal.npy")):
            self.normal_examples.add(Example(example[0], example[1]))
        for example in np.load(os.path.join(path_to_load, "spam.npy")):
            self.spam_examples.add(Example(example[0], example[1]))
        for example in np.load(os.path.join(path_to_load, "test.npy")):
            self.test_examples.add(Example(example[0], example[1]))

    def summary(self):
        print("Total normal: {}".format(len(self.normal_examples)))
        print("Total spam: {}".format(len(self.spam_examples)))
        print("Total test: {}, normal in test {}".format(len(self.test_examples),
                                                         len([1 for e in self.test_examples if e.label == 0])))


if __name__ == '__main__':
    demo_formatter = shijing888FormatData(
        "localhost", None,
        "../tmp/raw_data/normal",
        "../tmp/raw_data/spam",
        "../tmp/raw_data/test")
    demo_formatter.save_to_path("../tmp/data")
    demo_formatter.summary()

    # demo_formatter_load = shijing888FormatData(path_to_load="../tmp/data")
    # demo_formatter_load.summary()
