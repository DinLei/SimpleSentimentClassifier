#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/10/27 17:17
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import pandas as pd
import numpy as np


def train_and_test(text_data, prob=0.7, shuffle=True):
    import math
    import random
    from model_training.app_util.text_operator import TextOperation
    y_label = [x[1] for x in text_data]
    x_text_data = [TextOperation.sent_rebuild(x[0], read_emoji=False) for x in text_data]
    text_data_dict = {}
    for ind, label in enumerate(y_label):
        label = str(label)
        if label not in text_data_dict:
            text_data_dict[label] = []
        text_data_dict[label].append(x_text_data[ind])
    train_data = []
    test_data = []
    for label, records in text_data_dict.items():
        num = len(records)
        train_num = math.ceil(num * prob)
        tmp = list(zip(records, [label]*num))
        train_data.extend(tmp[:train_num])
        test_data.extend(tmp[train_num:])
    if shuffle:
        random.shuffle(train_data)
    return train_data, test_data, x_text_data


def load_data_and_labels(positive_data_file, negative_data_file,
                         balance=True, stop_words=None, rm_punctuation=False):
    from model_training.app_util.sampling import repetition_random_sampling
    from model_training.app_util.text_operator import TextOperation
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    positive_examples = list()
    positive_data = pd.read_table(positive_data_file)
    p_row, p_col = positive_data.shape
    for ind in range(p_row):
        row = positive_data.iloc[ind].to_dict()
        tmp = TextOperation.sent_rebuild(row["positive_reviews"],
                                         stop_words=stop_words,
                                         rm_punctuation=rm_punctuation)
        if tmp in ["no_english", "url"]:
            continue
        positive_examples.append(tmp)
    positive_examples = list(set(positive_examples))

    negative_examples = list()
    negative_data = pd.read_table(negative_data_file)
    n_row, n_col = negative_data.shape
    for ind in range(n_row):
        row = negative_data.iloc[ind].to_dict()
        tmp = TextOperation.sent_rebuild(
            row["negative_reviews"], stop_words=stop_words, rm_punctuation=rm_punctuation)
        if tmp in ["no_english", "url"]:
            continue
        negative_examples.append(tmp)
    negative_examples = list(set(negative_examples))

    len_pos = len(positive_examples)
    len_neg = len(negative_examples)
    if balance:
        len_max = max(len(positive_examples), len(negative_examples))
        if len_pos == len_max:
            add = repetition_random_sampling(negative_examples, (len_max - len_neg))
            negative_examples.extend(add)
        else:
            add = repetition_random_sampling(positive_examples, (len_max - len_pos))
            positive_examples.extend(add)

    print("negative examples:before sampling:{}, after sampling:{}...".format(len_neg, len(negative_examples)))
    print("positive examples:before sampling:{}, after sampling:{}...".format(len_pos, len(positive_examples)))
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def text2vec(text, t2v_model, stop_words=None, rm_punctuation=False):
    from model_training.app_util.text_operator import TextOperation
    if isinstance(text, str):
        text = TextOperation.sent_rebuild(
            text, stop_words=stop_words, rm_punctuation=rm_punctuation)
        if text == "url":
            text = "hate hate, dont like"
        return t2v_model.transform([text])
    elif isinstance(text, list):
        outcome = []
        for row in text:
            text = TextOperation.sent_rebuild(
                row, stop_words=stop_words, rm_punctuation=rm_punctuation)
            if text == "url":
                text = "hate hate, dont like"
            outcome.append(text)
        print(outcome)
        return t2v_model.transform(outcome)

