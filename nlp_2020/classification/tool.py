import torch.nn as nn
from torchtext.data import Field, LabelField, TabularDataset


#TODO: implement for train/dev/test
#TODO: replace `tokenize` by jieba tokenizer
def build_dataset(fpath, mode='train'):
    # For more info about torchtext.data,
    # turn to https://pytorch.org/text/data.html
    tokenize = lambda x: x.split()
    ID = Field(sequential=False, use_vocab=False)
    # NOTE: CATEGORY_CODE could be ignored
    CATEGORY_CODE = LabelField(sequential=False, use_vocab=False)
    CATEGORY = LabelField(sequential=False, use_vocab=False)
    NEWS = Field(
        sequential=True,
        use_vocab=False,
        tokenize=tokenize,
        include_lengths=True,
    )

    # Format of dataset:
    # 6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
    fields = [
        ('id', ID),
        ('category_code', CATEGORY_CODE),
        ('category', CATEGORY),
        ('news', NEWS),
        (None, None),
    ]

    # Since dataset is split by `_!_`.
    dataset = TabularDataset(
        fpath,
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '_!_'},
    )
    return (ID, CATEGORY, NEWS), dataset
