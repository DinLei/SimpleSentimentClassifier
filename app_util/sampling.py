#!/usr/bin/pyhton
# -*-coding:utf-8-*-
import random


def get_rand_num():
    return random.random()


def random_sampling(data_mat, number):
    try:
        slice1 = random.sample(data_mat, number)
        return slice1
    except Exception as e:
        print(e)
        print('sample larger than population')


def repetition_random_sampling(data_mat, number):
    sample = []
    len1 = len(data_mat) - 1
    for i in range(number):
        sample.append(data_mat[random.randint(0, len1)])
    return sample


def systematic_sampling(data_mat, number):
    length = len(data_mat)
    k = length / number
    sample = []
    i = 0
    if k > 0:
        while len(sample) != number:
            sample.append(data_mat[0 + i * k])
            i += 1
        return sample
    else:
        return random_sampling(data_mat, number)
