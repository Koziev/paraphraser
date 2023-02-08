"""
Файнтюн rugpt на датасете перефразировок с использованием GPT2DoubleHeadsModel (https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2DoubleHeadsModel)
Для проектов чатбота и генеративных стихов.

Используется датасет перефразировок из проекта чатбота с добавленными сэмплами проекта генеративных стихов.
В качестве дистракторов используем негативные примеры перефразировок из этого же датасета плюс рандомные выборки.

04.01.2023 Заранее подготовленный датасет загружаем из paraphrases.json (см. публичную версию https://huggingface.co/datasets/inkoziev/paraphrases)
13.01.2023 Для оценки близости текстов теперь используем LaBSE
05.02.2023 Метрики семантической и символьной похожести при финальной оценке разделены.
05.02.2023 Добавлен расчет BaryScore (см. https://arxiv.org/abs/2108.12463)
"""

import collections
import os
import json
import io
import random
import itertools
import re
import argparse

import numpy as np
import scipy
import tqdm
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import transformers
from transformers import AutoTokenizer
import sentence_transformers


num_distractors = 4
epochs = 1


def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2)) / float(1e-8 + len(shingles1 | shingles2))


def norm(s):
    s2 = s.lower().replace('ё', 'е')
    s3 = re.sub(r'[.,!?;:\- ]', '', s2)
    return s3


class Samples(object):
    def __init__(self, paraphrases, distractors):
        self.paraphrases = set(paraphrases)
        self.distractors = set(distractors)


# TODO: переделать на динамический подбор негативных сэмплов в каждом батче?
def load_samples(dataset_path, tokenizer):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    samples = [Samples(sample['paraphrases'], sample['distractors']) for sample in data]

    # Сразу отделим holdout для финальной оценки, и train/test для тренировки с early stopping.
    train_samples, test2_samples = sklearn.model_selection.train_test_split(samples, test_size=0.10, random_state=123456789)
    test_samples, eval_samples = sklearn.model_selection.train_test_split(test2_samples, test_size=1000, random_state=123456789)

    # НАЧАЛО ОТЛАДКИ
    #train_samples = train_samples[:1000]
    #test_samples = test_samples[:1000]
    #eval_samples = eval_samples[:100]
    # КОНЕЦ ОТЛАДКИ

    # В тех сэмплах, где не заданы негативные примеры, нам надо будет как-то подобрать их автоматически.
    # Сделаем это, рандомно выбирая фразы из пула всех фраз датасета.
    all_texts = set()
    positive_pairs = set()
    for sample in data:
        all_texts.update(sample['paraphrases'])
        all_texts.update(sample['distractors'])
        for phrase1, phrase2 in itertools.combinations(sample['paraphrases'], 2):
            n1 = norm(phrase1)
            n2 = norm(phrase2)
            positive_pairs.add((n1, n2))
            positive_pairs.add((n2, n1))
    all_texts = list(all_texts)

    # Конвертируем в сэмплы для обучения.

    num_candidates = num_distractors + 1

    datasets = {"train": collections.defaultdict(list), "valid": collections.defaultdict(list)}

    bos_token_id = tokenizer.encode('<s>')[0]
    eos_token_id = tokenizer.encode('</s>')[0]
    sep_token_id = tokenizer.encode('<sep>')[0]

    for dataset_name, dataset_samples0 in [('train', train_samples), ('valid', test_samples)]:
        dataset_samples = []

        for sample in tqdm.tqdm(dataset_samples0, desc='Compilation of ' + dataset_name, total=len(samples)):
            attractors = set(sample.paraphrases)
            distractors = set(sample.distractors)

            # Если заданных вручную негативных примеров мало, то добавим рандомных.
            while len(distractors) < num_distractors:
                distractor = random.choice(all_texts)
                if not any((norm(attractor), norm(distractor)) in positive_pairs for attractor in attractors):
                    distractors.add(distractor)

            # берем все сочетания правильных перефразировок, добавляя к каждой паре все негативные примеры.
            for phrase1, phrase2 in itertools.combinations(attractors, 2):
                if norm(phrase1) != norm(phrase2):
                    if len(distractors) > num_distractors:
                        distractors2 = sorted(distractors, key=lambda _: random.random())[:num_distractors]
                    else:
                        distractors2 = list(distractors)

                    dataset_samples.append((phrase1, distractors2 + [phrase2]))

        for src_text, paraphrases in tqdm.tqdm(dataset_samples, desc='Tokenization of ' + dataset_name, total=len(dataset_samples)):
            # только последний текст в paraphrases является валидной перефразировкой
            for j, paraphrase in enumerate(paraphrases):
                src_text_tokens = tokenizer.encode(src_text)
                paraphrase_tokens = tokenizer.encode(paraphrase)
                input_ids = [bos_token_id] + src_text_tokens + [sep_token_id] + paraphrase_tokens + [eos_token_id]

                if j == num_candidates - 1:
                    lm_labels = [-100] + [-100] * len(src_text_tokens) + [-100] + paraphrase_tokens + [eos_token_id]
                else:
                    lm_labels = [-100] * len(input_ids)

                # TODO: ПЕРЕДЕЛАТЬ НА ДВА СПЕЦТОКЕНА <prompt> и <output>
                # типом 1 помечаем начальный токен <s>, исходный текст и разделитель.
                # типом 0 помечаем токены перефразировки, затем <s>
                token_type_ids = [1] * (1 + len(src_text_tokens) + 1) + [0] * (len(paraphrase_tokens) + 1)

                mc_token_ids = len(input_ids) - 1  # классификатор срабатывает на последнем токене </s>

                datasets[dataset_name]['input_ids'].append(input_ids)
                datasets[dataset_name]['lm_labels'].append(lm_labels)
                datasets[dataset_name]['token_type_ids'].append(token_type_ids)
                datasets[dataset_name]['mc_token_ids'].append(mc_token_ids)
                datasets[dataset_name]['attention_mask'].append([1]*len(input_ids))

            datasets[dataset_name]['mc_labels'].append(num_candidates-1)  # у нас всегда последний вариант продолжения - корректный
            datasets[dataset_name]["n_candidates"] = num_candidates

        # выравниваем
        max_l = max(len(x) for x in datasets[dataset_name]["input_ids"])
        for input_name in ['input_ids', 'token_type_ids']:
            datasets[dataset_name][input_name] = [x + [tokenizer.pad_token_id] * (max_l - len(x)) for x in datasets[dataset_name][input_name]]
        datasets[dataset_name]['lm_labels'] = [x + [-100] * (max_l - len(x)) for x in datasets[dataset_name]['lm_labels']]
        datasets[dataset_name]['attention_mask'] = [x + [0] * (max_l - len(x)) for x in datasets[dataset_name]['attention_mask']]

    # финальное преобразование размерности и конвертация в тензор
    MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "attention_mask"]
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    return tensor_datasets, eval_samples


def train(model, device, train_generator, test_generator, optimizer, eval_steps):
    total_loss = 0
    for istep, (input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, attention_mask) in tqdm.tqdm(enumerate(train_generator, start=1), desc='Training', total=len(train_generator)):
        model.train()
        outputs = model(input_ids=input_ids.to(device),
                        labels=lm_labels.to(device),
                        #token_type_ids=token_type_ids.to(device),
                        mc_token_ids=mc_token_ids.to(device),
                        mc_labels=mc_labels.to(device),
                        attention_mask=attention_mask.to(device),
                        )
        loss = outputs.loss + outputs.mc_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if 0 == (istep % eval_steps):
            visualize(tokenizer, model, device, ['В лесу родилась ёлочка', 'Пора испить нектар любви',
                                                 'Мишка по лесу идет', 'Туман над озером клубится',
                                                 'Я иду, шагаю по Москве', 'Как хороши, как свежи были розы',
                                                 'У бурных чувств неистовый конец',
                                                 'Идет бычок, качается, вздыхает на ходу',
                                                 'Снег выпал в ноябре внезапно', 'Угрюмо кот взирает на елку',
                                                 'Стараюсь я не думать о грядущем', 'Одна голова - хорошо, а две - лучше'])

            print('Step {} evaluation...'.format(istep), end='', flush=True)
            test_loss = test(model, device, test_generator)
            print(' test_loss: {}'.format(test_loss))

    avg_train_loss = total_loss / len(train_generator)
    return avg_train_loss


def test(model, device, batch_generator):
    model.eval()
    total_loss = 0
    for input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, attention_mask in batch_generator:
        outputs = model(input_ids=input_ids.to(device),
                        labels=lm_labels.to(device),
                        #token_type_ids=token_type_ids.to(device),
                        mc_token_ids=mc_token_ids.to(device),
                        mc_labels=mc_labels.to(device),
                        attention_mask=attention_mask.to(device))
        loss = outputs.loss + outputs.mc_loss
        total_loss += loss.item()

    avg_test_loss = total_loss / len(batch_generator)
    return avg_test_loss


def generate_paraphrase(tokenizer, model, device, prompt):
    prompt_ids = tokenizer.encode(prompt)
    input_ids = tokenizer.encode('<s>') + prompt_ids + tokenizer.encode('<sep>')
    t_input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).to(device)
    outputs = model.generate(input_ids=t_input_ids,
                             # token_type_ids=None,
                             max_length=100,
                             temperature=1.0,
                             top_k=0,
                             top_p=0.85,
                             typical_p=None,
                             repetition_penalty=1.2,
                             do_sample=True,
                             num_return_sequences=1,
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
    print('-'*30 + ' VISUALIZATION ' + '-'*30)
    model.eval()
    for prompt in viz_prompts:
        generated_text = generate_paraphrase(tokenizer, model, device, prompt)
        print('{} ==> {}'.format(prompt, generated_text))
    print('-'*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paraphrase model finetuning')
    parser.add_argument('--model', type=str, default='sberbank-ai/rugpt3small_based_on_gpt2', help='Name or path of pretrained LM to be finetuned')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-6, help='Learning rate')

    args = parser.parse_args()

    proj_dir = os.path.expanduser('~/polygon/chatbot')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device={}'.format(device))

    output_dir = os.path.join(proj_dir, 'tmp', 'rugpt_paraphraser2')

    is_distributed = False  # TODO: сделать поддержку multi-gpu
    train_batch_size = args.batch_size
    valid_batch_size = args.batch_size
    eval_steps = 2000

    pretrained_model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    model = transformers.GPT2DoubleHeadsModel.from_pretrained(pretrained_model_name)
    model.to(device)

    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
    num_added_tokens = tokenizer.add_tokens(['<sep>'])  # добавляем спецтокен для отделения исходного текста и перефразировки
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    tokenizer.save_pretrained(output_dir)

    # Загружаем штатный датасет перефразировок - см. https://huggingface.co/datasets/inkoziev/paraphrases
    print('Loading dataset...')
    tensor_datasets, eval_samples = load_samples(os.path.join(proj_dir, 'tmp', 'paraphrases.json'), tokenizer)

    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, shuffle=(not is_distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=valid_batch_size, shuffle=False)

    print("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    print("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))

    #optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    #optimizer = optim.Adamax(model.parameters(), lr=1e-5)
    #optimizer = optim.RMSprop(model.parameters())
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    print(f'Start training with learning_rate={args.lr}')
    best_loss = np.inf
    for epoch in range(1, epochs+1):
        print('\n=== EPOCH {}/{} ==='.format(epoch, epochs))
        try:
            train_loss = train(model, device, train_loader, valid_loader, optimizer, eval_steps)
            print('\nTrain loss={}'.format(train_loss))

            test_loss = test(model, device, valid_loader)

            print('\nTest loss={}'.format(test_loss))
            #scheduler.step()

            print('Saving model to "{}"...'.format(output_dir))
            model.save_pretrained(output_dir)

            print('='*80)
        except KeyboardInterrupt:
            print('Training interrupted.')
            break

    # ---------------------------------------------------
    # Финальная оценка модели.
    # ---------------------------------------------------
    if len(eval_samples) > 0:
        print('*** EVALUATION ***')

        model.eval()

        # Нагенерируем перефразировки для всех тестовых сэмплов.
        # TODO: сделать пакетную генерацию в gpt, получение эмбеддинов в sbert батчами
        eval_texts = list(set(itertools.chain(*[sample.paraphrases for sample in eval_samples])))
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
        eval_texts = list(set(itertools.chain(*[sample.paraphrases for sample in eval_samples])))
        for eval_text, paraphrase in tqdm.tqdm(zip(eval_texts, eval_paraphrases), desc='Embedding similarity', total=len(eval_texts)):
            # Косинусная близость эмбеддингов
            vx = embedder.encode([eval_text, paraphrase], show_progress_bar=False, device="cuda" if use_cuda else "cpu").tolist()
            sim = 1.0 - scipy.spatial.distance.cosine(u=vx[0], v=vx[1])
            sem_sims.append(sim)

            # символьная похожесть
            j_sim = jaccard(eval_text, paraphrase, 3)
            char_sims.append(sim * (1.0 - j_sim))

        print('\nMean baryscore_W={}\n'.format(np.mean(wasserstein_dist)))
        print('\nMean semantic similarity = {}\nMean character similarity = {}'.format(np.mean(sem_sims), np.mean(char_sims)))
