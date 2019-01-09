import time
import keras
from keras_preprocessing import sequence

from ew.data_io import shijing888FormatData
from ew.models import EWSpamClassifierLSTM, EWSpamClassifier
import os

os.makedirs("./outputs/models", exist_ok=True)
os.makedirs("./outputs/logs", exist_ok=True)
os.makedirs("./tmp/data", exist_ok=True)

BERT_ENCODING_DIM = 768
N_CLASS = 2  # for spam
MAX_LEN = 20
BATCH_SIZE = 64
EPOCHS = 20

class EasyWay:
    def transform_data(self):
        data = shijing888FormatData(
            "localhost", None,
            "./tmp/raw_data/normal",
            "./tmp/raw_data/spam",
            "./tmp/raw_data/test")
        data.save_to_path("./tmp/data")
        data.summary()
        # !!! 注意 在开启 BERT-as-service 以后 tensorflow 不能再用GPU ?
        # 所以要先关掉BERT-as-service
        exit()

    def train(self):
        data = shijing888FormatData(path_to_load="./tmp/data")
        examples_train = data.get_examples_train()
        examples_train.shuffle()
        exp_name = str(time.time())
        model = EWSpamClassifierLSTM((MAX_LEN, BERT_ENCODING_DIM), exp_name, N_CLASS)
        print("start training...")
        model.train(examples_train, BATCH_SIZE, EPOCHS, MAX_LEN)

    def test(self):
        data = shijing888FormatData(path_to_load="./tmp/data")
        examples_test = data.test_examples
        classifier = EWSpamClassifier.load_model("./outputs/models/model_1547068994.801282/weights.13-0.03874.hdf5", 2)
        x_test, y_test = examples_test.to_keras_format(classifier.to_one_hot)
        x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

        score, acc = classifier.model.evaluate(x_test, y_test, batch_size=64)
        print('Test score:', score)
        print('Test accuracy:', acc)


if __name__ == '__main__':
    way = EasyWay()
    # 转换数据
    # way.transform_data()
    # 训练
    # way.train()
    # 测试
    way.test()
