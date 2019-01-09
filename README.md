# 用什么方法比较好
这好像是一个用BERT的训练好模型来做垃圾邮件分类的repository

## The EASY WAY
之所以说EASY是这个方法似乎可以不用管BERT具体的实现和架构
### 依赖
- [bert-as-service](https://github.com/hanxiao/bert-as-service)
    - 把bert当作服务(HTTP?)
    - 输入句子, 输出BERT的encoding
    - github上星星比较多
    - 如果是windows系统请看[如果使用Windows](#Windows)
    - 点击[这里](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)下载中文BERT的预先训练好的模型
        - 解压缩记录好位置
        - bert-serving-start -model_dir <模型的位置> -num_worker=1
        - windows不行就看[如果使用Windows](#Windows)
- Python >= 3.5
- Tensorflow >= 1.10


### Implementation
#### training
1. 原始数据
    - 每一个example是这样的pair:`(新闻, label)` (e.g. `label=1 is for spam`)
2. 预处理
    - 把一个新闻变成很多句子
        - <句子1><句子2><句子3>
    - 把<句子>丢到bert-as-service,得到这个句子的encoding比如`SE_1`
        - 不需要分词,全部丢就可以了
    - 所有一个新闻句子合在一起变成一个series
        - `NE_1 = [SE_1, SE_2, SE_3,...]`
    - 变成成一个新的训练example: 
        - `<NE_1, label>` 
    - 由于GPU一次只能载入一个模型(?)
        - 这边是先转换完保存在本地, 然后关掉`bert-serving-server`
3. 对于每一个<NE_1, label>,用LSTM来训练分类
    - 输入
        - 将文章 Padding 到长度 20 (即每篇文章20个句子,如果本身不满则补零)
        - input shape是(None, 20, 768) 其中768是下载的BERT模型的encoding的维度
        - 输出是one-hot: [1,0] 和 [0,1] 两类
    - 模型描述
        - 一个Bi-LSTM, 每个LSTM有256个hidden units, 且只取最后一个hidden state
        - 一个Dense层 hidden units的数量是500
        - 一个BatchNormalization层
        - 一个Dropout层
        - 最后经过一个输出是2的Dense (2是分类)
```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
masking_1 (Masking)          (None, 20, 768)           0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 512)               2099200   
_________________________________________________________________
dense_1 (Dense)              (None, 500)               256500    
_________________________________________________________________
batch_normalization_1 (Batch (None, 500)               2000      
_________________________________________________________________
dropout_1 (Dropout)          (None, 500)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 1002      
=================================================================
Total params: 2,358,702
Trainable params: 2,357,702
Non-trainable params: 1,000
```
        
4. 使用分类器来做序列的分类
    - `[SE_1, SE_2, SE_3,...]` -> `preferred_label`
    - `LSTM` or `bi-LSTM`

#### inference
1. 将新闻的每个句子得到一个encoding
    - 得到一个`NEWS = [SE_1, SE_2, SE_3,...]`
2. 将`NEWS`丢到训练好的模型得到结果
    - 这边还是要padding,理论上不用,input_1的shape[改成None即可](https://blog.csdn.net/weixin_40937909/article/details/80154879)

#### 表现
- 数据集
    - 找了半天的中文垃圾邮件分类数据集
        - 找到一个有数据有算法的[BayesSpam](https://github.com/shijing888/BayesSpam)
- 表现对比
    - [BayesSpam](https://github.com/shijing888/BayesSpam)的在测试集正确率是0.95
    - 我们的模型在EPOCH大约18左右可以到0.987 (没有fine tuned, 至少可以根据val_acc来选)
- 结论
    - 正确率高
    - 简单:不需要分词([BayesSpam](https://github.com/shijing888/BayesSpam)的停词似乎就在做这件事)
    - I want the date badly

### Windows
在这个时刻[2019/1/10 4:59]分,在*Windows10*上直接安装使用理论上会有些问题
- 如果提示找不到`bert-serving-start`

在当前目录使用以下指令:

  `python start-bert-as-service.py -model_dir ./tmp/chinese_L-12_H-768_A-12/ -num_worker=1`

- 如果提示`TypeError: can't pickle _thread.RLock objects`

修改 `<Python安装地址>/Lib/site-packages/bert_serving/server/helper.py` 下的`set_logger`函数(原因貌似是Win10多线程不能log到文件)
```python
# def set_logger(context, verbose=False):
#     logger = logging.getLogger(context)
#     logger.setLevel(logging.DEBUG if verbose else logging.INFO)
#     formatter = logging.Formatter(
#         '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
#         '%m-%d %H:%M:%S')
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
#     console_handler.setFormatter(formatter)
#     logger.handlers = []
#     logger.addHandler(console_handler)
#     return logger
class FakeLogger:
    def __init__(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        print(*args, **kwargs)

    def debug(self, *args, **kwargs):
        print(*args, **kwargs)


def set_logger(context, verbose=False):
    return FakeLogger()
```
## The HARD WAY
WHY?
