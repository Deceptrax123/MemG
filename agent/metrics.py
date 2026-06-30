import os
import json
import time
from datetime import datetime
from dateutil import parser as date_parser
from tqdm import tqdm
from collections import defaultdict
import torch
import collections
import re
import string
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(
        ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def calculate_bleu_scores(reference, prediction):
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]

    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0),
                    (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens,
                                  weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score

    return scores['bleu1']


def compute_f1(a_gold, a_pred):
    gold_norm = normalize_answer(a_gold)
    pred_norm = normalize_answer(a_pred)

    gold_toks = gold_norm.split()
    pred_toks = pred_norm.split()

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)

    f1 = (2 * precision * recall) / (precision + recall)
    return f1 * 100.0


def compute_bleu1(a_gold, a_pred):
    gold_toks, pred_toks = normalize_answer(
        a_gold).split(), normalize_answer(a_pred).split()
    if not pred_toks or not gold_toks:
        return 0.0
    num_same = sum((collections.Counter(gold_toks) &
                   collections.Counter(pred_toks)).values())
    p = num_same / len(pred_toks)
    bp = 1.0 if len(pred_toks) > len(gold_toks) else math.exp(
        1 - len(gold_toks) / len(pred_toks))
    return p * bp * 100


def compute_rouge_l(a_gold, a_pred):
    gold_toks, pred_toks = normalize_answer(
        a_gold).split(), normalize_answer(a_pred).split()
    if not pred_toks or not gold_toks:
        return 0.0
    m, n = len(gold_toks), len(pred_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold_toks[i-1] == pred_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p, r = lcs / n, lcs / m
    return (2 * p * r) / (p + r) * 100


def calculate_bert_scores(reference, prediction):
    try:
        P, R, F1 = bert_score([prediction], [reference],
                              lang='en', verbose=False)
        F1.item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")

        return 0.0
