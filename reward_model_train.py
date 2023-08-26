import os

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, context

from mindnlp.transforms import PadTransform
from mindnlp.transforms.tokenizers import BertTokenizer
from mindnlp.models import BertForSequenceClassification
from mindnlp._legacy.amp import auto_mixed_precision
from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy


class SentimentDataset:
    """Sentiment Dataset"""

    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            label, text_a = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)


def process_dataset(source, tokenizer, pad_value, max_seq_len=64, batch_size=32, shuffle=True):
    column_names = ["label", "text_a"]
    rename_columns = ["label", "input_ids"]

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    pad_op = PadTransform(max_seq_len, pad_value=pad_value)
    type_cast_op = transforms.TypeCast(mindspore.int32)

    # map dataset
    dataset = dataset.map(operations=[tokenizer, pad_op], input_columns="text_a")
    dataset = dataset.map(operations=[type_cast_op], input_columns="label")
    # rename dataset
    dataset = dataset.rename(input_columns=column_names, output_columns=rename_columns)
    # batch dataset
    dataset = dataset.batch(batch_size)

    return dataset


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
pad_value = tokenizer.token_to_id('[PAD]')

dataset_train = process_dataset(SentimentDataset("./data/0train.tsv"), tokenizer, pad_value)
dataset_val = process_dataset(SentimentDataset("./data/0dev.tsv"), tokenizer, pad_value)
dataset_test = process_dataset(SentimentDataset("./data/0test.tsv"), tokenizer, pad_value, shuffle=False)

# set bert config and define parameters for training
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model = auto_mixed_precision(model, 'O1')

loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=2e-5)

metric = Accuracy()

# define callbacks to save checkpoints
ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='bert_emotect', epochs=1, keep_checkpoint_max=2)
best_model_cb = BestModelCallback(save_path='checkpoint', ckpt_name='bert_emotect_best', auto_load=True)

trainer = Trainer(network=model, train_dataset=dataset_train,
                  eval_dataset=dataset_val, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer, callbacks=[ckpoint_cb, best_model_cb],
                  jit=True)
# start training
trainer.run('label')
evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
evaluator.run(tgt_columns="label")
dataset_infer = SentimentDataset("data/0infer.tsv")


def predict(text, label=None):
    label_map = {0: "口语化", 1: "书面的"}

    text_tokenized = Tensor([tokenizer.encode(text).ids])
    logits = model(text_tokenized)
    predict_label = logits[0].asnumpy().argmax()
    info = f"inputs: '{text}', predict: '{label_map[predict_label]}'"
    if label is not None:
        info += f" , label: '{label_map[label]}'"
    print(info)


from mindspore import Tensor

for label, text in dataset_infer:
    predict(text, label)
