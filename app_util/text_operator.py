#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/9/29 15:05
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import math
import re
from string import punctuation

import langid
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from model_training.app_util.review_translator import tran2eng

porter_stem = PorterStemmer()
lemmatizer = WordNetLemmatizer()


class TextOperation:
    """
    文本处理工具
    """

    @staticmethod
    def sent_rebuild(sentence, read_emoji=True, language_check=True, rm_digit=True,
                     to_stem=True, stop_words=None, rm_punctuation=False):
        """
        文本预处理——大写转小写&取词干&分词重组
        :return: 
        """
        assert isinstance(sentence, str)
        from model_training.app_util.terrible_situation import ugly_words
        for uk, uv in ugly_words.items():
            if uk in sentence:
                sentence = sentence.replace(uk, uv)
        if read_emoji:
            emo_search = re.findall(r"\[\[[^\].]+\]\]", sentence)
            for emo in emo_search:
                emo_mean = TextOperation.reading_emoji(emo)
                sentence = re.sub(emo, " {} ".format(emo_mean), sentence)
        else:
            sentence = re.sub(r"\[\[[^\].]+\]\]", "", sentence)

        if language_check:
            if not is_english(sentence):
                print("Translating is going...")
                source, sentence = tran2eng(sentence)
                print("{}->{}".format(source, sentence))
        # if re.search("https?\s?:\s?//[^\s]*", sentence):
        #     return "url"
        if rm_digit:
            sentence = re.sub(r"\d+", "", sentence)

        sent = clean_str(sentence)
        words = sent.split()
        words = [w.strip() for w in words]
        if rm_punctuation:
            words = [x for x in words if x not in punctuation]
        if stop_words:
            words = [w for w in words if w not in stop_words]
        if to_stem:
            words = [porter_stem.stem(wi) for wi in words]
        return " ".join(words)

    @staticmethod
    def n_gram(sentence, ngram_range=(1, 3)):
        """
        将一个字符串转成n-gram形式，单词级别
        """
        assert isinstance(sentence, str)
        sent_list = sentence.split()
        outcome = []
        for i in range(ngram_range[0], ngram_range[1] + 1):
            for j in range(len(sent_list) - i + 1):
                outcome.append(" ".join(sent_list[j: j + i]))
        return outcome

    @staticmethod
    def reading_emoji(emoji_code):
        from model_training.app_util.emoji_meaning import emoji_meaning
        check = re.match(r"^\[\[.{,20}\]\]$", emoji_code)
        assert check
        if emoji_code in emoji_meaning:
            return emoji_meaning[emoji_code]
        print("No this emoji recode")
        return ""


def text_vector_model(text_data, weighted='tf', binary=True,
                      ngram_range=(1, 3), max_df=0.99, min_df=1):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    assert weighted in {'tf', 'tf-idf'}
    if weighted == 'tf_idf':
        model = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df,
                                min_df=min_df, binary=binary)
    else:
        model = CountVectorizer(ngram_range=ngram_range, max_df=max_df,
                                min_df=min_df, binary=binary)
    return model.fit(text_data)


def get_model_need_data(data, model):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    assert isinstance(model, CountVectorizer) or isinstance(model, TfidfVectorizer)
    y_label = [x[1] for x in data]
    x_data = [x[0] for x in data]
    x_matrix = model.transform(x_data)
    return x_matrix, y_label


# modify_sigmoid和is_english函数用来筛选英文句子
def modify_sigmoid(x):
    return x * (1 / (1 + math.exp(-x)))


def is_english(sentence):
    is_en = 0
    nor_en = 0
    sents = re.split(r"[.?!;:,\-+#$…]+", sentence)
    for sent in sents:
        if langid.classify(sent)[0] == 'en':
            is_en += modify_sigmoid(len(sent))
        else:
            nor_en += modify_sigmoid(len(sent))
    if nor_en > is_en:
        return False
    return True


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", "nt", string)
    string = re.sub(r"n`t", "nt", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"x{2,}", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
