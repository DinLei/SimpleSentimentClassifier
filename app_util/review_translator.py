#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/14 10:53
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import time

from functools import wraps
from googletrans import Translator
translator = Translator(service_urls=[
    'translate.google.cn'
])


def time_elapse(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        cur = time.time()
        if not arg and not kwargs:
            func()
        elif arg and kwargs:
            func(*arg, **kwargs)
        elif arg:
            func(*arg)
        elif kwargs:
            func(**kwargs)
        print("耗时: %s 秒！" % (time.time()-cur))
    return wrapper


@time_elapse
def print_tran2eng(text, destination="en"):
    translations = translator.translate(text=text, dest=destination)
    for translation in translations:
        print(translation.origin, ' -> ', translation.text)
        print("\n")


def tran2eng(text, destination="en"):
    assert isinstance(text, list) or isinstance(text, str)
    try:
        translations = translator.translate(text=text, dest=destination)
        if isinstance(translations, list):
            return [(tr.origin, tr.text) for tr in translations]
        else:
            return translations.origin, translations.text
    except Exception as e:
        print(e)
        if isinstance(text, list):
            return [(None, tx) for tx in text]
        else:
            return None, text


if __name__ == "__main__":
    text_sam = [
        "Iga Viktoria Szczepanik super co ?",
        "Jane Kampen voor jou",
        "Estefania Parra Martinez",
        "bonjours journée porte ouvert sur les collines de mes dames XD"
    ]
    print_tran2eng(text_sam)

