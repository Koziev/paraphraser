"""
Эксперимент с файнтюном: токены исходного текста не включаем в backprop, присваивая соответствующим целям (labels) значение -100
"""

import os
import json
import io
import random
import itertools
import collections
import argparse

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
import sentence_transformers



def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2)) / float(1e-8 + len(shingles1 | shingles2))



def load_samples(dataset_path, tokenizer):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    train_data, eval_data = sklearn.model_selection.train_test_split(data, test_size=0.05, random_state=123456789)

    eval_texts = list(itertools.chain(*[sample['paraphrases'] for sample in eval_data]))

    samples = []
    for sample in train_data:
        for src_text, positive in itertools.combinations(sample['paraphrases'], r=2):
            input_tokens = tokenizer.encode(src_text)
            output_tokens = tokenizer.encode(positive)
            samples.append((input_tokens, output_tokens, src_text, positive))
            assert not any((t is None) for t in input_tokens)
            assert not any((t is None) for t in output_tokens)

    return samples, eval_texts


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

    parser = argparse.ArgumentParser(description='Paraphrase model finetuning')
    parser.add_argument('--model', type=str, default='sberbank-ai/rugpt3small_based_on_gpt2', help='Name or path of pretrained LM to be finetuned')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-6, help='Learning rate')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device={}'.format(device))

    pretrained_model_name = args.model

    output_dir = os.path.join(proj_dir, 'tmp', 'rugpt_paraphraser2')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
    num_added_tokens = tokenizer.add_tokens(['<sep>'])

    print('Loading dataset...')
    samples, eval_texts = load_samples(os.path.join(proj_dir, 'tmp', 'paraphrases.json'), tokenizer)
    train_samples, test_samples = sklearn.model_selection.train_test_split(samples, test_size=0.05, random_state=123456789)

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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_batch_size = args.batch_size
    valid_batch_size = args.batch_size
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=valid_batch_size)

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

    # ---------------------------------------------------
    # Финальная оценка модели.
    # ---------------------------------------------------
    if len(eval_texts) > 0:
        print('*** EVALUATION ***')

        model.eval()

        # Нагенерируем перефразировки для всех тестовых сэмплов.
        # TODO: сделать пакетную генерацию в gpt, получение эмбеддинов в sbert батчами
        eval_paraphrases = []
        for eval_text in tqdm.tqdm(eval_texts, desc='Paraphrasing'):
            paraphrase = generate_paraphrase(tokenizer, model, device, eval_text)
            eval_paraphrases.append(paraphrase)

        # Модель перефразировки нам больше не нужна.
        del model


        print('Running BaryScore metric calculations...')
        from nlg_eval_via_simi_measures.bary_score import BaryScoreMetric
        bary_scorer = BaryScoreMetric(model_name='sberbank-ai/ruBert-base', last_layers=5, use_idfs=True, sinkhorn_ref=0.01)
        hyps = eval_texts
        refs = eval_paraphrases
        wasserstein_dist = []
        with tqdm.tqdm(total=len(hyps)) as pbar:
            while len(hyps) > 0:
                batch_refs = refs[:train_batch_size]
                batch_hyps = hyps[:train_batch_size]
                bary_scorer.prepare_idfs(refs=batch_refs, hyps=batch_hyps)
                res = bary_scorer.evaluate_batch(batch_hyps=batch_hyps, batch_refs=batch_refs, show_tqdm=False)
                wasserstein_dist.extend(res['baryscore_W'])
                hyps = hyps[train_batch_size:]
                refs = refs[train_batch_size:]
                pbar.update(len(batch_hyps))
        del bary_scorer

        # Теперь оцениваем смысловую и буквальную близость текстов и их перефразировок.
        embedder_model_name = 'sentence-transformers/LaBSE'
        print('Calculate similarity metrics using "{}"...'.format(embedder_model_name))
        embedder = sentence_transformers.SentenceTransformer(embedder_model_name, device="cuda" if use_cuda else "cpu")
        sem_sims = []
        char_sims = []
        for eval_text, paraphrase in tqdm.tqdm(zip(eval_texts, eval_paraphrases), desc='Embedding similarity', total=len(eval_texts)):
            # Косинусная близость эмбеддингов
            vx = embedder.encode([eval_text, paraphrase], show_progress_bar=False, device="cuda" if use_cuda else "cpu").tolist()
            sim = 1.0 - scipy.spatial.distance.cosine(u=vx[0], v=vx[1])
            sem_sims.append(sim)

            # символьная похожесть
            j_sim = jaccard(eval_text, paraphrase, 3)
            char_sims.append(sim * (1.0 - j_sim))

        print('\n' + '='*80 + '\n')
        print('Mean baryscore_W          = {}'.format(np.mean(wasserstein_dist)))
        print('Mean semantic similarity  = {}'.format(np.mean(sem_sims)))
        print('Mean character similarity = {}'.format(np.mean(char_sims)))
