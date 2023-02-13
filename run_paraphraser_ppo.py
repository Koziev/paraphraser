"""
Небольшой эксперимент с использованием RL/PPO для файнтюна модели перефразировщика https://github.com/Koziev/paraphraser

"""
import traceback

import torch
import os
import io
import itertools
import json
import re
import random
import pickle

import tqdm
import sentence_transformers

import transformers
from transformers import GPT2Tokenizer

import pymorphy2

# https://github.com/lvwerra/trl
import trl
#from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
#from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
#from trl.ppo import PPOTrainer
#from trl.core import build_bert_batch_from_txt, listify_batch

from udpipe_parser import UdpipeParser


def tokenize(s):
    return re.split(r'[.,!?\- ;:]', s)


anaphoric_words = set('неё этого него этом нему он она оно они их его ее её ему ей ею им ими ею нем нём ней нею ними ним них это то этот тот эта та эти те этой этою той тою этих тех там тут здесь'.split())


bad_words = set('неё этого него все этом нему увы даже как впрочем хотя хоть если ежели коль вот уж уже ужель же ли ль ка бы или либо но да он она оно они их его ее её ему ей ею им ими ею нем ним нём ней нею ними них это то этот тот эта та эти те этой этою той тою этих тех'.split())

def stop_word(s):
    if len(s) < 2:
        return True
    if s in bad_words:
        return True
    return False


def word_sim(s1, s2):
    """Близость двух мешков слов"""
    tokens1 = tokenize(s1)
    tokens2 = tokenize(s2)

    tokens1 = set(t.lower().replace('ё', 'е') for t in tokens1 if not stop_word(t))
    tokens2 = set(t.lower().replace('ё', 'е') for t in tokens2 if not stop_word(t))

    return float(len(tokens1 & tokens2)) / float(len(tokens1 | tokens2) + 1e-6)



def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1, shingle_len)
    shingles2 = ngrams(s2, shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


def is_proper_name(word):
    if len(word) > 1 and word[0].lower() != word[0]:
        if re.match(r'[A-Z][a-z]+', word) is not None:
            return True

        for tagset in morph.parse(word):
            if any((tag in ['Name', 'Sgtm', 'Geox', 'Patr', 'Orgn']) for tag in tagset.tag._grammemes_tuple):
                return True

    return False


def find_pred_lemma(text):
    """Ищем сказуемое через синтаксический разбор"""
    for parsing in parser.parse_text(text):
        for token in parsing:
            if token.deprel == 'root':
                return token.lemma
    return '<UNK>'


def is_cyr(s):
    return re.match(r'^[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+$', s, re.I)


def shingles3(text):
    sx = set()
    for word in tokenize(text):
        if is_cyr(word):
            text2 = '[' + word + ']'
            sx.update(c1+c2+c3 for c1, c2, c3 in zip(text2, text2[1:], text2[2:]))

    return sx


def find_defects(seed_text, text):
    num_defects = 0

    seed_words = tokenize(seed_text)
    seed_props = [word for word in seed_words if is_proper_name(word)]

    # Если после перефразировки глагольное сказуемое не изменилось (та же лемма), то считаем дефектом.
    seed_verb = find_pred_lemma(seed_text)

    seed_unk3 = [c3 for c3 in shingles3(seed_text) if c3 not in char_3grams]

    # Проверяем, что имена собственные не пропали.
    # TODO: отслеживать также названия месяцев, числа и числительные, названия дней недели.
    para_words = tokenize(text)
    text_props = [word for word in para_words if is_proper_name(word)]
    if any((prop not in text_props) for prop in seed_props):
        num_defects += 20

    text_verb = find_pred_lemma(text)
    if text_verb == seed_verb:
        # Глагольное сказуемое не поменялось - это плохо.
        num_defects += 2

    # Если в генерации появились неизвестные символьные триграммы
    text_unk3 = [c3 for c3 in shingles3(text) if c3 not in char_3grams and c3 not in seed_unk3]
    num_defects += len(text_unk3)

    # Если появилось сочетание латиницы и кириллицы
    num_defects += len(re.findall(r'[абвгдеёжзийклмнопрстуфхцчшщъыьэюя][a-z]', text, flags=re.I))
    num_defects += len(re.findall(r'[a-z][абвгдеёжзийклмнопрстуфхцчшщъыьэюя]', text, flags=re.I))

    # Если символ повторяется 4 и более раз подряд
    num_defects += 2.0 * len(re.findall(r'(.)\1{3,}', text))

    # если 2-грамма повторяется 3 и более раз подряд
    #  Я не знаю ответьте очень надо пжпжажпжажажажажпжпжпжпжпжпжпжпжпжпжпжпжпжпжпжппжжппжпжжпжппдпжжппжпжп№ппжпжпжпжпжпжпжпжжппжпжпжпжп
    num_defects += 3.0 * len(re.findall(r'(.{2})\1{3,}', text))

    # если 3-грамма повторяется 3 и более раз подряд
    num_defects += 10.0 * len(re.findall(r'(.{3})\1{3,}', text))

    # если есть слово длиннее 40 символов
    num_defects += 10.0 * sum(len(t)>=35 for t in tokenize(text))

    # Появление в генерации всяких тегов типа <s>
    num_defects += 10.0 * len(re.findall(r'<[a-z]+>', text))

    # Если в перефразировке появляется анафористическое слово типа ОН, которого не было в исходном тексте,
    # назначаем штраф.
    for kw in anaphoric_words:
        r = '\\b' + kw + '\\b'
        if re.search(r, seed_text, flags=re.I) is None and re.search(r, text, flags=re.I) is not None:
            num_defects += 5.0

    return num_defects


def reward_fun(src_text, para_text):
    wsim = word_sim(src_text, para_text)

    seed_v = critic.encode([src_text])[0]
    otext_v = critic.encode([para_text])[0]
    sim = sentence_transformers.util.cos_sim(a=seed_v, b=otext_v).item()

    if sim >= 0.0:
        reward = (0.9 - wsim) * sim
    else:
        reward = sim

    defects = find_defects(src_text, para_text)
    reward = max(-1.0, reward - defects/10.0)

    assert (-1.0 <= reward <= 1.0)
    return reward


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    proj_dir = '.'

    # Сюда будем писать файлы с логами
    output_dir = os.path.join(proj_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    parser = UdpipeParser()
    parser.load('./data/udpipe_syntagrus.model')

    morph = pymorphy2.MorphAnalyzer()

    # Используем базу символьных 3-грамм для обнаружения кривой генерации.
    ngrams_path = './data/char_3grams.pkl'
    if os.path.exists(ngrams_path):
        with open(ngrams_path, 'rb') as f:
            char_3grams = pickle.load(f)

    assert find_defects('Кошка ловит мышку', 'Она её ловит') > 0.0
    assert find_defects('Кошка ловит мышку', 'мышку преследует кошка') == 0.0

    critic_model_name = 'inkoziev/sbert_synonymy'
    print('Loading critic from "{}"...'.format(critic_model_name))
    critic = sentence_transformers.SentenceTransformer(critic_model_name, device=device)

    #paraphraser_model_name = os.path.join(proj_dir, 'tmp', 'rugpt_paraphraser2')
    paraphraser_model_name = 'inkoziev/paraphraser'
    print('Loading generative model from "{}"...'.format(paraphraser_model_name))
    gpt2_model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(paraphraser_model_name).to(device)
    gpt2_model_ref = trl.AutoModelForCausalLMWithValueHead.from_pretrained(paraphraser_model_name).to(device)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(paraphraser_model_name)

    # Загружаем короткие предложения из файла. Там собраны предложения, в которых
    # встречается новая для перефразировщика лексика.
    with open('./data/paraphrase_seeds.json', 'r') as f:
        seeds = json.load(f)

    batch_size = 1

    # initialize trainer
    ppo_config = trl.PPOConfig(batch_size=1, forward_batch_size=batch_size, learning_rate=1e-6)
    ppo_trainer = trl.PPOTrainer(config=ppo_config, model=gpt2_model, ref_model=gpt2_model_ref, tokenizer=gpt2_tokenizer)

    # В этот файл будем записывать выдачу reward model на каждом шаге, чтобы потом визуализировать ход эксперимента.
    wrt_ppo = io.open(os.path.join(output_dir, 'paraphraser_ppo.csv'), 'w', encoding='utf-8')
    wrt_ppo.write('step\treward\tsrc_text\tparaphrase_text\n')

    wrt_positives = io.open(os.path.join(output_dir, 'paraphraser_ppo.positives.txt'), 'w', encoding='utf-8')
    wrt_negatives = io.open(os.path.join(output_dir, 'paraphraser_ppo.negatives.txt'), 'w', encoding='utf-8')

    sep_token_id = gpt2_tokenizer.encode('<sep>')[0]

    counter = 0

    input_tokens_batch = []
    output_tokens_batch = []
    reward_batch = []

    for seed_text in tqdm.tqdm(seeds):
        encoded_prompt = gpt2_tokenizer.encode('<s>' + seed_text + '<sep>', add_special_tokens=False, return_tensors="pt").to(device)
        output_sequences = gpt2_model.generate(input_ids=encoded_prompt,
                                               max_length=100,
                                               typical_p=0.85,
                                               top_k=0,
                                               top_p=1.0,
                                               do_sample=True,
                                               num_return_sequences=5,
                                               pad_token_id=gpt2_tokenizer.pad_token_id)

        texts = set()
        positives = []
        negatives = []

        for o in output_sequences:
            output_tokens = o.tolist()

            # отрезаем затравку
            output_tokens = output_tokens[output_tokens.index(sep_token_id)+1:]

            if gpt2_tokenizer.pad_token_id in output_tokens:
                output_tokens = output_tokens[:output_tokens.index(gpt2_tokenizer.pad_token_id)]

            if gpt2_tokenizer.eos_token_id in output_tokens:
                output_tokens = output_tokens[:output_tokens.index(gpt2_tokenizer.eos_token_id)]

            otext = gpt2_tokenizer.decode(output_tokens, clean_up_tokenization_spaces=True)
            if otext not in texts:
                texts.add(otext)
                counter += 1

                reward = reward_fun(seed_text, otext)

                if reward > 0:
                    positives.append(otext)
                elif reward < 0:
                    negatives.append(otext)

                input_tokens_batch.append(encoded_prompt.cpu()[0].tolist())
                output_tokens_batch.append(output_tokens + [gpt2_tokenizer.eos_token_id])
                reward_batch.append(reward)

                if len(input_tokens_batch) >= batch_size:
                    query_tensors = [torch.LongTensor(input_tokens_batch).squeeze().to(device)]
                    response_tensors = [torch.LongTensor(output_tokens_batch).squeeze().to(device)]
                    reward_tensors = [torch.tensor(r) for r in reward_batch]

                    try:
                        # train model with ppo
                        train_stats = ppo_trainer.step(queries=query_tensors, responses=response_tensors, scores=reward_tensors)
                        for reward in reward_batch:
                            wrt_ppo.write('{}\t{}\t{}\t{}\n'.format(counter, reward, seed_text, otext))
                        wrt_ppo.flush()
                    except ValueError as err:
                        print('\nDEBUG@292 ValueError\n{}'.format(err))
                        print(traceback.format_exc())
                        exit(0)

                    input_tokens_batch = []
                    output_tokens_batch = []
                    reward_batch = []

        if positives:
            wrt_positives.write('{}\n'.format(seed_text))
            for otext in positives:
                wrt_positives.write('{}\n'.format(otext))
            wrt_positives.write('\n\n\n')
            wrt_positives.flush()

        if negatives:
            wrt_negatives.write('{}\n'.format(seed_text))
            for otext in negatives:
                wrt_negatives.write('(-) {}\n'.format(otext))
            wrt_negatives.write('\n\n\n')
            wrt_negatives.flush()

    wrt_ppo.close()
    wrt_positives.close()
    wrt_negatives.close()

    print('All done :)')
