from mindnlp.transforms.tokenizers import BertTokenizer
from mindnlp.models import BertForSequenceClassification
from mindnlp._legacy.amp import auto_mixed_precision
from mindspore import Tensor
import jieba
import time


def predict(text):
    text = jiebaSplit(text)
    label_map = {0: "口语化", 1: "书面的"}

    text_tokenized = Tensor([tokenizer.encode(text).ids])
    logits = model(text_tokenized)
    predict_label = logits[0].asnumpy().argmax()
    info = f"\ninputs: '{text}', \npredict: '{label_map[predict_label]}'\n"
    print(info)


def jiebaSplit(str):
    # 精确模式
    seg_list = jieba.cut(str, cut_all=False)
    res = ' '.join(seg_list)
    return res


import os

if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    model = auto_mixed_precision(model, 'O1')

    os.system('cls')
    print("运行测试：")
    predict("家人们谁懂啊咱就是说一整个无语住了")

    while (True):
        print("输入exit或键入ctrl+C退出")
        print("请输入句子：")
        sentence = input()
        if sentence == "exit":
            break
        predict(sentence)

    print("程序结束，谢谢使用！")
    time.sleep(1)