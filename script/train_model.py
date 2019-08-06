#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/10/27 15:40
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import sys
sys.path.append("..")
from model_training.app_util.data_helper import *
from model_training.app_util.text_operator import *
from model_training.app_util.io_util import save_as_pickle
from model_training.app_util.classification_algorithms import *


positive_data_file = "../data/positive_comments.txt"
negative_data_file = "../data/negative_comments.txt"

stopwords = set([x.strip() for x in open("../app_util/english", "r").readlines()])
text, label = load_data_and_labels(
    positive_data_file, negative_data_file, stop_words=stopwords, rm_punctuation=True)

test_data = list(zip(text, label.tolist()))

train_raw, test_raw, total_texts = train_and_test(test_data, prob=1.0)

tv_model = text_vector_model(total_texts, weighted="tf-idf", max_df=0.90, min_df=1, ngram_range=(1, 3))

x_train, y_train = get_model_need_data(train_raw, tv_model)
x_test, y_test = get_model_need_data(test_raw, tv_model)


nb_model = naive_bayes_classifier(x_train, y_train)
# target_names = ["[1, 0]", "[0, 1]"]
# nb_model.predict_log_proba()
save_as_pickle(nb_model, "sentiment_recognizer.model", "../model")
save_as_pickle(tv_model, "construct_vector.model", "../model")


