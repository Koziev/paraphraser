"""
Эксперимент с файнтюном: токены исходного текста не включаем в backprop, присваивая соответствующим целям (labels) значение -100
"""

import os
import json
import io
import random
import itertools

import numpy as np
import tqdm
import sklearn.model_selection
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer



def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2)) / float(1e-8 + len(shingles1 | shingles2))


def load_samples(dataset_path, tokenizer):
    samples = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        for sample in data:
            for src_text, positive in itertools.combine(sample['paraphrases']):
                input_tokens = tokenizer.encode(src_text)
                output_tokens = tokenizer.encode(positive)
                samples.append((input_tokens, output_tokens, src_text, positive))
                assert not any((t is None) for t in input_tokens)
                assert not any((t is None) for t in output_tokens)

    return samples


class FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 0
        self.samples = []

        self.bos_token_id = tokenizer.encode('<s>')[0]
        self.eos_token_id = tokenizer.encode('</s>')[0]
        self.sep_token_id = tokenizer.encode('<sep>')[0]
        self.pad_token_id = tokenizer.encode('<pad>')[0]

        for src_ids, output_ids, _, _ in samples:
            input_ids = [self.bos_token_id] + src_ids + [self.sep_token_id] + output_ids + [self.eos_token_id]

            token_type_ids = [1] * (1 + len(src_ids)) + [0] + [0]*(len(output_ids) + 1)

            # Токены затравки дают label=-100
            # AHTUNG: сдвигать цепочку не надо, это происходит ВНУТРИ КЛАССА МОДЕЛИ (в forward)
            labels = [-100]*(1 + len(src_ids) + 1) + output_ids + [self.eos_token_id]

            self.samples.append((input_ids, token_type_ids, labels))
            self.max_len = max(self.max_len, len(input_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        input_ids, token_type_ids, labels = self.samples[index]
        npad = self.max_len - len(input_ids)
        input_ids = input_ids + npad*[self.pad_token_id]
        token_type_ids = token_type_ids + [0] * npad
        labels = labels + [-100] * npad
        return torch.LongTensor(input_ids), torch.LongTensor(token_type_ids), torch.LongTensor(labels)


def train(model, device, batch_generator, optimizer):
    total_loss = 0
    for input_ids, token_type_ids, labels in tqdm.tqdm(batch_generator, desc='Training', total=len(batch_generator)):
        model.train()
        t_input_ids = input_ids.to(device)
        #t_token_type_ids = token_type_ids.to(device)
        t_labels = labels.to(device)
        outputs = model(input_ids=t_input_ids,
                        #token_type_ids=t_token_type_ids,
                        labels=t_labels,
                        attention_mask=None)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(batch_generator)
    return avg_train_loss


def test(model, device, batch_generator):
    model.eval()
    total_loss = 0
    for input_ids, token_type_ids, labels in tqdm.tqdm(batch_generator, desc='Validation', total=len(batch_generator)):
        t_input_ids = input_ids.to(device)
        #t_token_type_ids = token_type_ids.to(device)
        t_labels = labels.to(device)
        outputs = model(input_ids=t_input_ids,
                        #token_type_ids=t_token_type_ids,
                        labels=t_labels,
                        attention_mask=None)
        loss = outputs.loss
        total_loss += loss.item()

    avg_test_loss = total_loss / len(batch_generator)
    return avg_test_loss


def mean_pooling(model_output, attention_mask):
    """ Mean Pooling - Take attention mask into account for correct averaging """
    token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_paraphrase(tokenizer, model, device, prompt):
    prompt_ids = tokenizer.encode(prompt)
    input_ids = tokenizer.encode('<s>') + prompt_ids + tokenizer.encode('<sep>')
    t_input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).to(device)

    outputs = model.generate(input_ids=t_input_ids,
                             #token_type_ids=None,
                             max_length=100,
                             temperature=1.0,
                             top_k=0,
                             top_p=0.85,
                             typical_p=None,
                             repetition_penalty=1.2,
                             do_sample=True,
                             num_return_sequences=2,
                             pad_token_id=tokenizer.pad_token_id,
                           )

    o1 = outputs[0]
    generated_ids = o1.detach().cpu().tolist()[len(input_ids):]
    generated_text = tokenizer.decode(generated_ids)
    if '</s>' in generated_text:
        generated_text = generated_text[:generated_text.index('</s>')]
    return generated_text


def visualize(tokenizer, model, device, viz_prompts):
    """ Визуализация генерации. """
    print('\n'+ '-'*30 + ' VISUALIZATION ' + '-'*30)
    model.eval()
    for prompt in viz_prompts:
        generated_text = generate_paraphrase(tokenizer, model, device, prompt)
        print('{} ==> {}'.format(prompt, generated_text))
    print('-'*80)


if __name__ == '__main__':
    proj_dir = os.path.expanduser('~/polygon/chatbot')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device={}'.format(device))

    pretrained_model_name = 'sberbank-ai/rugpt3large_based_on_gpt2'

    output_dir = os.path.join(proj_dir, 'tmp', 'rugpt_paraphraser2')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
    num_added_tokens = tokenizer.add_tokens(['<sep>'])

    print('Loading dataset...')
    samples = load_samples(os.path.join(proj_dir, 'tmp', 'paraphrases.json'), tokenizer)
    train_samples, test2_samples = sklearn.model_selection.train_test_split(samples, test_size=0.10, random_state=123456789)
    test_samples, eval_samples = sklearn.model_selection.train_test_split(test2_samples, test_size=0.50, random_state=123456789)
    # eval_samples будут нужны для финальной оценки качества. Модель не будет видеть их в ходе тренировки и тестов.

    print('Train samples: {}, test samples: {}'.format(len(train_samples), len(test_samples)))

    train_dataset = FinetuneDataset(train_samples, tokenizer)
    test_dataset = FinetuneDataset(test_samples, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    model.to(device)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    tokenizer.save_pretrained(output_dir)

    #optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    #optimizer = optim.Adamax(model.parameters(), lr=1e-5)
    #optimizer = optim.RMSprop(model.parameters())
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    batch_size = 10
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    epochs = 1

    best_loss = np.inf
    for epoch in range(1, epochs+1):
        print('\n=== EPOCH {}/{} ==='.format(epoch, epochs))
        try:
            train_loss = train(model, device, train_generator, optimizer)
            print('\nTrain loss={}'.format(train_loss))

            visualize(tokenizer, model, device, ['Мишка по лесу идет', 'Туман над озером клубится',
                                                 'Я иду, шагаю по Москве', 'Как хороши, как свежи были розы',
                                                 'У бурных чувств неистовый конец', 'Идет бычок, качается, вздыхает на ходу'])

            test_loss = test(model, device, test_generator)
            print('\nTest loss={}'.format(test_loss))
            #scheduler.step()
            print('='*80)
        except KeyboardInterrupt:
            print('Training interrupted.')
            break

    # Финальная оценка
    if len(eval_samples) > 0:
        model.eval()
        embedder_model_name = 'sberbank-ai/sbert_large_mt_nlu_ru'
        print('Calculate final metrics using "{}"...'.format(embedder_model_name))

        print('Loading BERT model "{}"...'.format(embedder_model_name))
        bert_tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)
        bert_model = transformers.AutoModel.from_pretrained(embedder_model_name)
        bert_model.eval()
        bert_model.to(device)

        sims = []
        for _, _, src_text, _ in tqdm.tqdm(eval_samples, desc='Evaluation'):
            paraphrase = generate_paraphrase(tokenizer, model, device, src_text)
            encoded_input = bert_tokenizer([src_text, paraphrase], padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded_input = encoded_input.to(device)
            with torch.no_grad():
                model_output = bert_model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            vx = embeddings.detach().cpu().tolist()
            sim = 1.0 - scipy.spatial.distance.cosine(u=vx[0], v=vx[1])
            j_sim = jaccard(src_text, paraphrase, 3)
            sims.append(sim * (1.0 - j_sim))

        print('Mean quality: {}'.format(np.mean(sims)))
