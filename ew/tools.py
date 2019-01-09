from bert_serving.client import BertClient
import re

from ew.defs import Example, Examples


class BERTEncoder:
    def __init__(self, ip, sentence_delimiters):
        """

        :param ip:
        :type ip:
        :param sentence_delimiters:
        :type sentence_delimiters:
        """
        self.bc = BertClient(ip=ip)
        self.sentence_delimiters = sentence_delimiters  # type: list

    def encode_sentences(self, sentences):
        return self.bc.encode(sentences)

    def encode_news(self, news):
        for delimiter in self.sentence_delimiters:
            # 把所有分隔符都换成 "|" 效率很差 但是简单...
            news = news.replace(delimiter, "|")
        sentences = news.split("|")
        return self.encode_sentences(sentences)


def preprocessing(ip, sentence_delimiters, raw_examples):
    """

    :param ip:
    :type ip: str
    :param sentence_delimiters:
    :type sentence_delimiters: list
    :param raw_examples:
    :type raw_examples: list of RawExample
    :return:
    :rtype: list
    """
    encoder = BERTEncoder(ip, sentence_delimiters)
    return Examples(
        [Example(encoder.encode_news(raw_example.news), raw_example.label) for raw_example in raw_examples])


def demo_BERTEncoder():
    sentence_delimiters = ["？ "]
    demo_encoder = BERTEncoder("localhost", sentence_delimiters)
    print(demo_encoder.encode_sentences(['hey you', 'whats up?', '你好么？', '我 还 可以']))
    print(demo_encoder.encode_sentences(['hey you', 'whats up?', '你好么？', '我 还 可以'])[1].shape)

    print(demo_encoder.encode_news('你好么？ 我还可以'))


if __name__ == '__main__':
    demo_BERTEncoder()
    # demo_Preprocessor()
