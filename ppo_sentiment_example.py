import os
import time
import random

from tqdm import tqdm
import numpy as np
from mindspore import Tensor, ops

from mindnlp.models.bert import BertForSequenceClassification
from mindnlp.transforms.tokenizers import BertTokenizer
from mindnlp.transforms.tokenizers import GPT2Tokenizer

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer

from iSummaryWriter import iSummaryWriter

writer = iSummaryWriter(log_path='./logs', log_name='PPO-Sentiment-Zh')
config = {
    "model_name": 'uer/gpt2-chinese-cluecorpussmall',
    "steps": 20000,
    "batch_size": 128,
    "forward_batch_size": 16,
    "ppo_epochs": 4,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
    "gen_len": 16,
    "save_freq": 5,
    'save_dir': 'checkpoints/ppo_sentiment_gpt'
}

# prompt池
prompts = [
    '刚收到货，感觉',
    '这部电影很',
    '说实话，真的很',
    '这次购物总的来说体验很'
]

# 情感分类模型
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
bert_model.load_parameter_slice('')
# bert_model = BertForSequenceClassification.load('./')

# 文本生成模型
gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

# RL Trainer
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)
total_ppo_epochs = int(np.ceil(config["steps"] / config['batch_size']))

for epoch in tqdm(range(total_ppo_epochs)):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {
        'tokens': [],
        'query': []
    }
    for _ in range(config['batch_size']):
        random_prompt = random.choice(prompts)  # 随机选择一个prompt
        tokens = gpt2_tokenizer.encode(random_prompt)
        batch['tokens'].append(tokens)
        batch['query'].append(random_prompt)
    query_tensors = [Tensor(t).long() for t in batch["tokens"]]

    t = time.time()
    response_tensors = []
    for i in range(config['batch_size']):
        gen_len = config['gen_len']
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),  # generate()用于直接生成token_id
                                       max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time() - t

    t = time.time()
    texts = [q + r for q, r in zip(batch['query'], batch['response'])]  # 计算正向/负向情感得分
    bert_outputs = bert_model(texts)
    rewards = []
    for output in bert_outputs:
        if output['label'] == 'positive (stars 4 and 5)':
            rewards.append(output['score'])
        elif output['label'] == 'negative (stars 1, 2 and 3)':
            rewards.append(1 - output['score'])
        else:
            raise ValueError(f"错误的推理结果{output['label']}.")
    rewards = Tensor(rewards)  # 将正向情感的得分作为生成得分
    timing['time/get_sentiment_preds'] = time.time() - t

    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)  # PPO Update
    timing['time/optimization'] = time.time() - t

    timing['time/epoch'] = time.time() - t0  # logging
    logs.update(timing)
    logs.update(stats)
    mean = ops.ReduceMean()
    logs['env/reward_mean'] = mean(rewards).numpy()
    logs['env/reward_std'] = ops.std(rewards).numpy()
    logs['env/reward_dist'] = rewards.numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")

    print('Random Sample 5 text(s) of model output:')
    for i in range(5):  # 随机打5个生成的结果
        print(f'{i + 1}. {random.choice(texts)}')

    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    for k, v in timing.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalar('ppo/loss/policy', stats['ppo/loss/policy'], epoch)
    writer.add_scalar('ppo/loss/value', stats['ppo/loss/value'], epoch)
    writer.add_scalar('ppo/policy/entropy', stats['ppo/policy/entropy'], epoch)
    writer.add_scalar('ppo/policy/policykl', stats['ppo/policy/policykl'], epoch)
    writer.record()

    if epoch % config['save_freq'] == 0:
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        cur_save_path = os.path.join(
            config['save_dir'], f'model_{epoch}_{round(float(logs["env/reward_mean"]), 2)}'
        )
        ppo_trainer.model.save_pretrained(cur_save_path)
        ppo_trainer.tokenizer.save_pretrained(cur_save_path)
